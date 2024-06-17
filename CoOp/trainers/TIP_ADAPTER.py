import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch import autograd

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


def load_clip(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    clip_model, _ = clip.load(backbone_name)

    return clip_model


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def pre_load_features(clip_model, images, target):
    with torch.no_grad():
        images, label = images.cuda(), target.cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features, label


def build_cache_model(cfg, clip_model, train_loader_cache):
    if not os.path.exists('cache/' + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots" + "seed" + str(cfg.SEED) + ".pt"):
        cache_keys = []
        cache_values = []
        augment_epoch = cfg.TRAINER.TIP_ADAPTER.AUGMENT_EPOCH
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(augment_epoch):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, augment_epoch))
                for i, batch in enumerate(tqdm(train_loader_cache)):
                    images, target = batch['img'].cuda(), batch['label'].cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, 'cache/' + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots" + "seed" + str(cfg.SEED) + ".pt")
        torch.save(cache_values, 'cache/' + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots" + "seed" + str(cfg.SEED) + ".pt")

    else:
        cache_keys = torch.load('cache/' + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots" + "seed" + str(cfg.SEED) + ".pt")
        cache_values = torch.load('cache/' + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots" + "seed" + str(cfg.SEED) + ".pt")

    return cache_keys, cache_values


@TRAINER_REGISTRY.register()
class TIP_ADAPTER(TrainerX):
    # Tip-Adapter: https://arxiv.org/abs/2207.09519

    def __init__(self, cfg, pre_train_info=None):
        super().__init__(cfg, pre_train_info)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TIP_ADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.alpha = cfg.TRAINER.TIP_ADAPTER.ALPHA
        self.beta = cfg.TRAINER.TIP_ADAPTER.BETA

        classnames = self.dm.dataset.classnames_o

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip(cfg)
        clip_model.eval()

        if cfg.TRAINER.TIP_ADAPTER.PREC == "fp32" or cfg.TRAINER.TIP_ADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = clip_model
        self.model.to(self.device)

        # Textual features
        print("Getting textual features as CLIP's classifier.")
        self.clip_weights = clip_classifier(classnames, IMAGENET_TEMPLATES_SELECT, self.model)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        self.cache_keys, self.cache_values = build_cache_model(cfg, self.model, self.train_loader_x)

        self.adapter = nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(self.model.dtype).cuda()
        self.adapter.weight = nn.Parameter(self.cache_keys.t().clone().detach())
        n_feat = 1024 if cfg.MODEL.BACKBONE.NAME == "RN50" else 512
        n_feat = 768 if cfg.MODEL.BACKBONE.NAME == "ViT-L-14" else n_feat

        self.optim = build_optimizer(self.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter", self.adapter, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.TIP_ADAPTER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        input, label, domain_label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.TIP_ADAPTER.PREC
        if prec == "amp":
            with autocast():
                with torch.no_grad():
                    image_features = self.model.encode_image(input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                affinity = self.adapter(image_features)
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
                clip_logits = 100. * image_features @ self.clip_weights
                tip_logits = clip_logits + cache_logits * self.alpha
                loss = F.cross_entropy(tip_logits, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = self.adapter(image_features)
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
            clip_logits = 100. * image_features @ self.clip_weights

            tip_logits = clip_logits + cache_logits * self.alpha
            loss = F.cross_entropy(tip_logits, label)

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(tip_logits, label)[0].item(),
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

    def parse_batch_test(self, batch):
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
            if '46' not in self.cfg.DATASET.NAME and 'PACS_' not in self.cfg.DATASET.NAME and 'VLCS_' not in self.cfg.DATASET.NAME:
                data_loader = self.test_loader
                print("Do evaluation on test set (semantic-in test)")
            else:
                data_loader = loader
                print("Do energy estimation on semantic-shifted ood test set")

        if split == 'test':
            data_loader = self.test_loader
            print("Do evaluation on {} set".format(split))

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_train(batch)
            self.adapter.eval()

            with torch.no_grad():
                image_features = self.model.encode_image(input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            affinity = self.adapter(image_features)
            test_beta = self.beta
            cache_logits = ((-1) * (test_beta - test_beta * affinity)).exp() @ self.cache_values
            clip_logits = 100. * image_features @ self.clip_weights
            tip_logits = clip_logits + cache_logits * self.alpha

            energy = torch.logsumexp(tip_logits, dim=-1)

            if '46' not in self.cfg.DATASET.NAME and 'PACS_' not in self.cfg.DATASET.NAME and 'VLCS_' not in self.cfg.DATASET.NAME:
                self.evaluator.process(tip_logits, label)
                self.evaluator.EE([tip_logits, energy], label)
            elif split != "extra_test":
                self.evaluator.process(tip_logits, label)
                self.evaluator.EE([tip_logits, energy], label, val_mod=True)
            else:
                self.evaluator.EE([tip_logits, energy], label, val_mod=False)

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

    def model_inference(self, input, domain_label, label):
        self.adapter.eval()
        with torch.no_grad():
            image_features = self.model.encode_image(input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        test_beta = self.beta
        affinity = self.adapter(image_features)
        cache_logits = ((-1) * (test_beta - test_beta * affinity)).exp() @ self.cache_values

        clip_logits = 100. * image_features @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha

        energy = torch.logsumexp(tip_logits, dim=-1)
        return [tip_logits, image_features, energy]

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

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_train(batch)
            output = self.model_inference(input, domain, label)
            self.evaluator.process(output, label)
            self.evaluator.EE(output, label, val_mod=True)

        results = self.evaluator.evaluate()

        data_loader = self.test_loader
        print("Do evaluation on Closed-set OOD test set")
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_train(batch)
            output = self.model_inference(input, domain, label)
            self.evaluator.EE(output, label, val_mod=True)

        energy_theta = results["energy_threshold"]
        energy_list_cood = results["energy_list"].tolist()

        split = "extra_test"
        data_loader = self.ood_test_loader
        print("Do evaluation on {} set".format(split))
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_train(batch)
            output = self.model_inference(input, domain, label)
            self.evaluator.EE(output, label, val_mod=False)

        results = self.evaluator.evaluate(energy_ref=energy_theta)

        energy_list_sood = results["energy_list"].tolist()
        AUROC = roc_auc_score(np.array([1] * len(energy_list_cood) + [0] * len(energy_list_sood)),
                              np.array(energy_list_cood + energy_list_sood))
        print('AUROC:{}'.format(AUROC))

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
            input, label, domain = self.parse_batch_train(batch)
            output = self.model_inference(input, domain, label)

            self.evaluator.process(output, label)
            self.evaluator.EE(output, label, val_mod=True)

        results = self.evaluator.evaluate()
        energy_theta = results["energy_threshold"]
        energy_theta_list = results["energy_list"]

        split = "extra_test"
        data_loader = self.ood_test_loader
        print("Do evaluation on {} (open-set OOD) set".format(split))
        self.evaluator.reset()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_train(batch)
            output = self.model_inference(input, domain, label)

            self.evaluator.EE(output, label, val_mod=False)

        results = self.evaluator.evaluate(energy_ref=energy_theta, energy_ref_list=energy_theta_list)

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)


