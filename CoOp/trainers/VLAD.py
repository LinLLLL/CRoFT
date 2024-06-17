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
    'ColoredMNIST': 'a photo of a digit {}.',
    'ColoredCatsDogs': 'a photo of a {}.'
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == 'ViT-L-14':
#        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(backbone_name, pretrained='laion400m_e31')
        model = torch.jit.load('/home/zl/DATA/ViT-L-14.pt', map_location='cpu')
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
    def __init__(self, c_in, reduction=4, non_linear=False, TEXT_ENHANCED=False):
        super(Adapter, self).__init__()
        self.TEXT_ENHANCED = TEXT_ENHANCED
        if non_linear:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                # nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(c_in // reduction, c_in, bias=False),
                # nn.Dropout(0.3),
                nn.ReLU(inplace=True),
            )
            # self._initialize_weights(self.fc)
            if TEXT_ENHANCED:
                self.fc_t = nn.Sequential(
                    nn.Linear(c_in, c_in // reduction, bias=False),
                    # nn.Dropout(0.3),
                    nn.ReLU(inplace=True),
                    nn.Linear(c_in // reduction, c_in, bias=False),
                    # nn.Dropout(0.3),
                    nn.ReLU(inplace=True),
                )
                # self._initialize_weights(self.fc_t)

        else:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in, bias=False)
            )
            # self._initialize_weights(self.fc)

            if TEXT_ENHANCED:
                self.fc_t = nn.Sequential(
                    nn.Linear(c_in, c_in, bias=False)
                )
                # self._initialize_weights(self.fc_t)

    def forward(self, x, t=None):
        x = self.fc(x)
        if t is not None and self.TEXT_ENHANCED:
            t = self.fc_t(t)
            return x, t
        else:
            return x


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


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
        if cfg.TRAINER.VLAD.TEXT_ENHANCED:
            self.adapter = Adapter(n_feat, 4, non_linear, True).to(clip_model.dtype)
        else:
            self.adapter = Adapter(n_feat, 4, non_linear, False).to(clip_model.dtype)

        self.ratio = cfg.TRAINER.VLAD.ratio
        self.ratio_text = cfg.TRAINER.VLAD.ratio_text

    def forward(self, image):
        text_features = self.text_encoder()

        image_features = self.image_encoder(image.type(self.dtype))
        if self.cfg.TRAINER.VLAD.TEXT_ENHANCED:
            x, text_features_enhanced = self.adapter(image_features, text_features)
        else:
            x = self.adapter(image_features)

        ratio = self.ratio
        image_features = ratio * x + (1 - ratio) * image_features  ###

        if self.cfg.TRAINER.VLAD.TEXT_ENHANCED:
            text_features_enhanced = self.adapter(text_features)
            text_features = self.ratio_text * text_features_enhanced + (1 - self.ratio_text) * text_features
            # text_features_enhanced = text_features_enhanced / text_features_enhanced.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image_features_sum = image_features_sum / image_features_sum.norm(dim=-1, keepdim=True)  ###

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()   ###

        return [logits, image_features, text_features]


class DIVLoss(nn.Module):
    def __init__(self, num_classes):
        super(DIVLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = 100.

    def forward(self, feature, query, target):
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(query, p=2, dim=1))  # (N, C)
        label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
        pred = torch.masked_select(pred.transpose(1, 0), label)  # N, # get positive query
        pred = pred.unsqueeze(1)  # (N, 1)

        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
        feature_s = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        # thres = torch.sum(feature_s * label_matrix) / torch.sum(label_matrix)  # get the average similarity of positive pairs
        feature_s = feature_s * ~label_matrix  # get negative matrix

        sorted = torch.argsort(feature_s, descending=True)
        nn_label_matrix = (sorted < 10).to(feature.dtype)

        pred_nn = F.linear(feature, F.normalize(query, p=2, dim=1)).repeat(feature.shape[0], 1, 1)# (N, N, C)
        label = label.transpose(1,0).repeat(1, feature.shape[0]).view(feature.shape[0], feature.shape[0], -1)
        pred_nn = torch.masked_select(pred_nn, label).view(feature.shape[0], -1)  # (N, N)
        pred_nn = torch.mean(torch.mm(pred_nn, 1. * nn_label_matrix), dim=1, keepdim=True) # (N, 1)

        logits = torch.cat([pred, pred_nn], dim=1)  # (N, 2)
        label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)

        return loss

@TRAINER_REGISTRY.register()
class VLAD(TrainerX):
    """ VLAD """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_nei = cfg.TRAINER.VLAD.knn  # 10 for CCD, 15 for PACS and VLCS, 5 for CelebA
        # PACS and VLCS rn50 linear adapter, other non_linear adapter

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model, cfg.TRAINER.VLAD.non_linear_adapter)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

        self.divloss = DIVLoss(2)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output, latent, text_feature = self.model(image)
        loss_CE = F.cross_entropy(output, label)
        loss_al_d, loss_al_a = self.regularization(latent, text_feature, label)
        # loss_al_d = self.divloss(latent, text_feature, label)
        loss = loss_CE + self.cfg.TRAINER.VLAD.lambda2 * loss_al_d + self.cfg.TRAINER.VLAD.lambda1 * loss_al_a
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item(),
#            'loss_a': loss_al_a.item(),
#            'loss_d': loss_al_d.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def regularization(self, latent, ek, y):
        distance = torch.mm(latent, latent.t())
        y = y.reshape(y.shape[0], -1)
        y1 = y.repeat(1, y.shape[0])
        y2 = y.t().repeat(y.shape[0], 1)
        lc = 1 * (y1 == y2)
        sort0 = torch.argsort(distance * lc, descending=True)

#        index0 = torch.where(lc == 1)
        index0 = torch.where((1 * (sort0 < self.n_nei)) & (lc==1))
        sort1 = torch.argsort(distance * (1 - lc), descending=True)
        index1 = torch.where((1 * (sort1 < self.n_nei)) & (lc == 0))
        pos = torch.sum(torch.exp(torch.diagonal(
            torch.mm(latent[index0[1]], ek[y[index0[0]]].squeeze(1).t())))) / len(index0[0])
        neg = torch.sum(torch.exp(torch.diagonal(
            torch.mm(latent[index1[1]], ek[y[index1[0]]].squeeze(1).t())))) / len(index0[0])
        neg1 = torch.sum(torch.exp(torch.mm(latent[index0[1]], ek.t()))) / len(index0[0])

        loss_al_d = -torch.log(pos / neg)
        loss_al_a = -torch.log(pos / neg1)
        print(pos.item(), neg.item(), neg1.item())

        return loss_al_d, loss_al_a

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
