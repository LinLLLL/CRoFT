import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, n_outputs, cfg):
        super(MLP, self).__init__()
        self.mlp_depth = cfg.mlp_depth
        self.mlp_dropout = cfg.mlp_dropout
        self.mlp_width = cfg.mlp_width
        self.input = nn.Linear(n_inputs, self.mlp_width)
        self.dropout = nn.Dropout(self.mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(self.mlp_width, self.mlp_width)
            for _ in range(self.mlp_depth-2)])
        self.output = nn.Linear(self.mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPLCLIP.N_CTX
        ctx_init = cfg.TRAINER.DPLCLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.DPLCLIP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPLCLIP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, domainnames, clip_model, sentence_prompt=False):
        super().__init__()
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.domainnames = domainnames
        self.cfg = cfg
        self.clip_model = clip_model

        self.n_feat = 1024 if cfg.MODEL.BACKBONE.NAME == "RN50" else 512
        self.n_feat = 768 if cfg.MODEL.BACKBONE.NAME == "ViT-L-14" else self.n_feat

        self.network = MLP(self.n_feat, 512 * self.cfg.TRAINER.DPLCLIP.N_CTX, cfg)
        self.network.to(self.dtype)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.network.apply(init_weights)

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * self.cfg.TRAINER.DPLCLIP.N_CTX)

        if sentence_prompt:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.classnames]
        else:
            classnames = [name.replace('_', ' ') for name in self.classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)

        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        self.register_buffer('token_suffix', embedding[:, self.cfg.TRAINER.DPLCLIP.N_CTX + 1:, :])  # CLS, EOS

    def forward(self, image_all):
        image_all = [image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1]) for image in image_all]
        image_feature_sort = [self.image_encoder(image.type(self.dtype)) for image in image_all]

        # print('image_features.shape', image_features.shape)
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] ->
        #  [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_feature_sort]  # concat image_feature from 3 domains
        image_features = torch.cat(image_feature_sort)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.classnames), dim=0) for feature in
                                 mean_domain_features]

        #  generate text_feature from domain_feature. text_features.size = [3, 7, 1024]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        energy = torch.logsumexp(logits, 1)

        return [logits, image_features, energy]

    def _get_text_features(self, domain_feature):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.cfg.TRAINER.DPLCLIP.N_CTX, 512)

        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)

        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(
            dim=-1)] @ self.clip_model.text_projection
        return text_features


@TRAINER_REGISTRY.register()
class DPLCLIP(TrainerX):
    """Context Optimization (DPLCLIP).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPLCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        domainnames = self.dm.dataset.alldomain

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DPLCLIP.PREC == "fp32" or cfg.TRAINER.DPLCLIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, domainnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "DPL" not in name:
                param.requires_grad_(False)
        for name, param in self.model.network.named_parameters():
            param.requires_grad_(True)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.network, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("DPL", self.model.network, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DPLCLIP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain_label = self.parse_batch_train(batch)
        image_all, label_all = self.sort_images(image, domain_label, label)
        label_new = torch.cat([l for l in label_all])

        prec = self.cfg.TRAINER.DPLCLIP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image_all)[0]
                loss = F.cross_entropy(output, label_new)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image_all)[0]
            loss = F.cross_entropy(output, label_new)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label_new)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain_label = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain_label = domain_label.to(self.device)

        return input, label, domain_label

    def sort_images(self, image, domain_label, label):
        image_sort = []
        label_sort = []
        for i in torch.unique(domain_label).cpu().numpy().tolist():
            image_sort.append(image[domain_label == i])
            label_sort.append(label[domain_label == i])
        return image_sort, label_sort

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None, UQ_theta=None, energy_theta=None, loader=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            # print(split)
            print("Do evaluation on {} set".format(split))

        if split == "extra_test" and self.ood_test_loader is not None:
            if 'BN' not in self.cfg.DATASET.NAME and '46' not in self.cfg.DATASET.NAME and 'PACS_' not in self.cfg.DATASET.NAME and 'VLCS_' not in self.cfg.DATASET.NAME:
                data_loader = self.test_loader
                print("Do evaluation on test set (semantic-in test)")
            else:
                data_loader = loader  # i.e., self.ood_test_loader
                print("Do energy estimation on semantic-shifted ood test set")

        if split == 'test':
            data_loader = self.test_loader
            print("Do evaluation on {} set".format(split))

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label_new = torch.cat([l for l in label_all])

            if '46' not in self.cfg.DATASET.NAME and 'PACS_' not in self.cfg.DATASET.NAME and 'VLCS_' not in self.cfg.DATASET.NAME:
                output = self.model_inference(image_all)[0]
                self.evaluator.process(output, label_new)
                self.evaluator.UQ(output, label_new)
            elif split != "extra_test":
                output = self.model_inference(image_all)
                self.evaluator.process(output, label_new)
                self.evaluator.EE(output, label_new, val_mod=True)
            else:
                output = self.model_inference(image_all)
                self.evaluator.EE(output, label_new, val_mod=False)

        # results = self.evaluator.evaluate(theta_test=theta) if theta is not None else self.evaluator.evaluate()
        if UQ_theta is not None and energy_theta is None:
            results = self.evaluator.evaluate(UQ_ref=UQ_theta)
        elif energy_theta is not None and split != "test":
            results = self.evaluator.evaluate(energy_ref=energy_theta)
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0], results["UQ_threshold"], results["energy_threshold"]

    @torch.no_grad()
    def eval_ood_detection(self):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        split = "val"
        data_loader = self.val_loader
        print("Do evaluation on {} set for energy threshold".format(split))
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label_new = torch.cat([l for l in label_all])
            output = self.model_inference(image_all)

            self.evaluator.process(output, label_new)
            self.evaluator.EE(output, label_new, val_mod=True)

        results = self.evaluator.evaluate()
        energy_theta = results["energy_threshold"]
        energy_theta_list = results["energy_list"]

        split = "extra_test"
        data_loader = self.ood_test_loader
        print("Do evaluation on {} (open-set OOD) set".format(split))
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label_new = torch.cat([l for l in label_all])
            output = self.model_inference(image_all)

            self.evaluator.EE(output, label_new, val_mod=False)

        # results = self.evaluator.evaluate(energy_ref=energy_theta)
        results = self.evaluator.evaluate(energy_ref=energy_theta, energy_ref_list=energy_theta_list)

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

    @torch.no_grad()
    def load_2dtsne(self, file_name="npy"):
        from sklearn import manifold, datasets
        from sklearn.decomposition import PCA
        import numpy as np
        from sklearn.metrics import roc_auc_score
        self.set_model_mode("eval")
        self.evaluator.reset()

        split = "val"
        data_loader = self.val_loader
        print("Do evaluation on {} set for energy threshold".format(split))
        image_features = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label = torch.cat([l for l in label_all])
            output = self.model_inference(image_all)

            self.evaluator.process(output, label)
            self.evaluator.EE(output, label, val_mod=True)

        results = self.evaluator.evaluate()
        energy_theta = results["energy_threshold"]

        data_loader = self.test_loader
        print("Do evaluation on Closed-set OOD test set")
        image_features = []
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label = torch.cat([l for l in label_all])
            output = self.model_inference(image_all)
            self.evaluator.EE(output, label, val_mod=True)

        # energy_theta = results["energy_threshold"]
        energy_list_cood = results["energy_list"].tolist()

        split = "extra_test"
        data_loader = self.ood_test_loader
        print("Do evaluation on {} set".format(split))
        image_features = []
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            image_all, label_all = self.sort_images(input, domain_label, label)
            label = torch.cat([l for l in label_all])
            output = self.model_inference(image_all)

            self.evaluator.EE(output, label, val_mod=False)
            # image_features.append(output[1].detach().cpu().numpy())

        results = self.evaluator.evaluate(energy_ref=energy_theta)

        energy_list_sood = results["energy_list"].tolist()
        AUROC = roc_auc_score(np.array([1] * len(energy_list_cood) + [0] * len(energy_list_sood)),
                              np.array(energy_list_cood + energy_list_sood))
        print('AUROC:{}'.format(AUROC))

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0], results["UQ_threshold"], results["energy_threshold"]



