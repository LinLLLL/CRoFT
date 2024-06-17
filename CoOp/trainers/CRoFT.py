import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import autograd

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'CelebA': 'a person with {}.',
    'VLCS': 'a photo of a {}.',
    'PACS': 'a photo of a {}.',
    'PACS_food101': 'a photo of a {}.',
    'PACS_DTD': 'a photo of a {}.',
    'PACS_Caltech101': 'a photo of a {}.',
    'VLCS_food101': 'a photo of a {}.',
    'VLCS_DTD': 'a photo of a {}.',
    'VLCS_Caltech101': 'a photo of a {}.',
    'ImageNet46': 'a photo of a {}.',
    'ImageNetOOD46': 'a photo of a {}.',
    'ColoredMNIST': 'a photo of a digit {}.',
    'ColoredCatsDogs': 'a photo of a {}.',
    'VLCS_flowers': 'a photo of a {}.',
}

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == 'ViT-L-14':
        model = torch.jit.load('model/ViT-L-14.pt', map_location='cpu')
        model = clip.build_model(model.state_dict())
    elif backbone_name == 'ViT-B-16':
        print("****************Loading ViTB16 from openCLIP****************")
        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(backbone_name, pretrained='laion400m_e31')
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location='cpu').eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location='cpu')

        model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in):
        super(Adapter, self).__init__()

        self.fc_i = nn.Parameter(torch.eye(c_in))
        self.fc_t = nn.Parameter(torch.eye(c_in))

    def forward(self, x, t):

        x = x @ self.fc_i.t()
        t = t @ self.fc_t.t()
        return x, t


class TextEncoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model.to('cuda')
        self.dtype = clip_model.dtype

    def forward(self):
        if 'ImageNet' in self.cfg.DATASET.NAME:
            with torch.no_grad():
                clip_weights = []
                for classname in self.classnames:
                    # Tokenize the prompts
                    classname = classname.replace('_', ' ')
                    texts = [t.format(classname) for t in IMAGENET_TEMPLATES_SELECT]
                    texts = clip.tokenize(texts).cuda()
                    # prompt ensemble for ImageNet
                    class_embeddings = self.clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)

                x = torch.stack(clip_weights, dim=1).cuda().t()
        else:
            with torch.no_grad():
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME] 
                prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompts = prompts.to('cuda')
                text_features = self.clip_model.encode_text(prompts)
                x = text_features
        return x


class OODImgGen(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.img2img_cov = nn.Parameter(torch.eye(c_in))

    def forward(self, text_features_id):
        image_feature_cov = text_features_id @ self.img2img_cov
        return image_feature_cov


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, non_linear):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        n_feat = 1024 if cfg.MODEL.BACKBONE.NAME == "RN50" else 512
        n_feat = 768 if cfg.MODEL.BACKBONE.NAME == "ViT-L-14" else n_feat
        self.adapter = Adapter(n_feat).to(clip_model.dtype)
        self.generator = OODImgGen(n_feat).to(clip_model.dtype)

        self.ratio = cfg.TRAINER.CRoFT.ratio
        self.ratio_text = cfg.TRAINER.CRoFT.ratio_text
        self.lambda1 = cfg.TRAINER.CRoFT.lambda1
        self.lambda2 = cfg.TRAINER.CRoFT.lambda2
        self.text_features = self.text_encoder()

    def feature_perturbation(self, image_features, text_features, id_label):
        label_matrix = id_label.unsqueeze(1) == id_label.unsqueeze(0)  # (N, N)
        image_features_cov = self.generator(image_features) + image_features
        image_features_cov = image_features_cov / (1e-6 + image_features_cov.norm(dim=-1, keepdim=True))
        logits_cov = self.logit_scale.exp() * image_features_cov @ text_features.t()
        # keep the semantic information by maintaining classification acc
        loss_ce = F.cross_entropy(logits_cov, id_label)
        # push away from the ID image features
        loss_disalign = (label_matrix * (image_features_cov @ image_features.t())).mean()
        loss = loss_ce + self.lambda1 * loss_disalign

        return loss, image_features_cov

    def EDR(self, image_features, image_features_cov_zs, text_features, target, theta, image_features_zs, text_features_zs):
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
        loss_disalign = (label_matrix * (image_features_cov_zs @ image_features_zs.t())).mean()
        image_features_cov = self.ratio * image_features_cov_zs + (1 - self.ratio) * (image_features_cov_zs @ theta[0])

        if self.cfg.TRAINER.CRoFT.WCCF and "VLCS" not in self.cfg.DATASET.NAME and self.cfg.TRAINER.CRoFT.gen_step > 0:
            logits_cov = self.logit_scale.exp() * image_features_cov @ text_features.t()
            loss1 = F.cross_entropy(logits_cov, target) - self.lambda1 * loss_disalign
            image_features_zs = torch.vstack([image_features_zs, image_features_cov_zs])
            image_features = torch.vstack([image_features, image_features_cov])
        else:
            logits_cov = self.logit_scale.exp() * image_features_cov_zs @ text_features.t()
            loss1 = F.cross_entropy(logits_cov, target)

        delta_theta = 0.0001 * torch.ones_like(theta[0]).cuda()
        delta_theta.requires_grad_(True)
        pred = torch.exp(image_features @ text_features.t())
        image_features_new = image_features_zs @ (theta[0] + delta_theta)
        text_features_new = text_features_zs @ (theta[1] + delta_theta)
        image_features_new = self.ratio * image_features_new + (1 - self.ratio) * image_features_zs
        text_features_new = self.ratio_text * text_features_new + (1 - self.ratio_text) * text_features_zs
        image_features_new = image_features_new / (1e-6 + image_features_new.norm(dim=-1, keepdim=True))
        text_features_new = text_features_new / (1e-6 + text_features_new.norm(dim=-1, keepdim=True))
        pred_new = torch.exp(image_features_new @ text_features_new.t())
        pred_delta = pred_new - pred

        if "46" in self.cfg.DATASET.NAME:
            # for setup1 imagenet46
            grad = autograd.grad(pred_delta.sum(), delta_theta, create_graph=True)[0]
            loss2 = (grad ** 2).mean()  # gradient of the proposed energy function
        else:
            # for setup2 cross-dataset open ood detection
            grad = autograd.grad(pred_delta.mean(), delta_theta, create_graph=True)[0]
            loss2 = (grad ** 2).sum()  # gradient of the proposed energy function

        loss = loss1 + self.lambda2 * loss2 if self.cfg.TRAINER.CRoFT.WCCF else self.lambda2 * loss2

        return loss

    def forward(self, image):
        text_features = self.text_features
        image_features = self.image_encoder(image.type(self.dtype))
        text_features_zs = text_features / (1e-6 + text_features.norm(dim=-1, keepdim=True))
        image_features_zs = image_features / (1e-6 + image_features.norm(dim=-1, keepdim=True))
        x, t = self.adapter(image_features, text_features)
        image_features = self.ratio * x + (1 - self.ratio) * image_features  # image feature adapter
        text_features = self.ratio_text * t + (1 - self.ratio_text) * text_features  # text feature adapter

        image_features = image_features / (1e-6 + image_features.norm(dim=-1, keepdim=True))
        text_features = text_features / (1e-6 + text_features.norm(dim=-1, keepdim=True))

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        energy = torch.logsumexp(logits, 1)
        # energy.requires_grad_(requires_grad=True)

        return [logits, image_features, text_features, image_features_zs, text_features_zs, energy]


@TRAINER_REGISTRY.register()
class CRoFT(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames_o

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model, cfg.TRAINER.CRoFT.non_linear_adapter)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name and 'generator' not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give adapter and generator to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.optim_generator = build_optimizer(self.model.generator, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.sched_generator = build_lr_scheduler(self.optim_generator, cfg.OPTIM)
        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)
        self.register_model('generator', self.model.generator, self.optim_generator, self.sched_generator)

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output, image_features, text_features, image_features_zs, text_features_zs, energy = self.model(image)
        if self.cfg.TRAINER.CRoFT.WCCF:
            for i in range(self.cfg.TRAINER.CRoFT.gen_step):
                feature_perturbation_loss, image_features_cov = \
                    self.model.feature_perturbation(image_features, text_features, label)
                self.model_backward_and_update(feature_perturbation_loss)

        image_features_cov = self.model.feature_perturbation(image_features, text_features, label)[1]

        theta = [self.model.adapter.fc_i, self.model.adapter.fc_t]

        EDR_loss = self.model.EDR(image_features, image_features_cov, text_features, label, theta, image_features_zs, text_features_zs)

        loss_CE = F.cross_entropy(output, label)
        loss = loss_CE + EDR_loss
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
