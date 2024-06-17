import numpy as np
import os.path as osp
from torch.nn import functional as F
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._UQ_c = []
        self._UQ_e = []
        self.loss_test = []
        self.confidence_gt, self.detect, self.res_all = [], [], []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._UQ_c = []  # collect the logits of correct classified val samples
        self._UQ_e = []  # collect the logits of incorrect classified val samples
        self.loss_test = []  # collect the cross-entropy loss of correct classified val samples
        self.confidence_gt, self.detect, self.res_all = [], [], []  # to plot AUC-ROC curve in Bayes-CAL (AAAI2023)

        self._EE = []  # collect the energy estimation of correct classified val samples
        self.EE_res_all = [] # collect the energy estimation of all val samples
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo or mo[0]: model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo[0] if isinstance(mo, list) else mo
        # mo = mo[0] if len(mo) ==5 else mo
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)
                
    def UQ(self, mo, gt):
        # mo or mo[0]: model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo[0] if isinstance(mo, list) else mo
        # mo = mo[0] if len(mo) ==5 else mo
        pred = mo.max(1)[1]
        softmax_logits = torch.nn.functional.softmax(mo, 1)
        
        res = softmax_logits.max(1)[0]
        matches = pred.eq(gt).float()
        error_idx = torch.where(matches==0)[0].cpu().numpy().tolist()
        correct_idx = torch.where(matches==1)[0].cpu().numpy().tolist()

        self._UQ_e.extend(res[error_idx].cpu().numpy().tolist())
        self._UQ_c.extend(res[correct_idx].cpu().numpy().tolist())
        loss_test = F.cross_entropy(mo[error_idx], gt[error_idx])
        self.loss_test.append(loss_test.item())
        self.confidence_gt.extend([1 if i in correct_idx else 0 for i in range(res.shape[0])])
        self.res_all.extend(res.cpu().numpy().tolist())

    def EE(self, mo, gt, val_mod=False):
        assert isinstance(mo, list)  # ensure you have output the energy value
        res = mo[-1]  # summing over the inner product between image feature and ID test features
        if val_mod:
            pred = mo[0].max(1)[1]
            matches = pred.eq(gt).float()
            # error_idx = torch.where(matches == 0)[0].cpu().numpy().tolist()
            correct_idx = torch.where(matches == 1)[0].cpu().numpy().tolist()
            res = res[correct_idx]
        self._EE.extend(res.cpu().numpy().tolist())
        self.EE_res_all.extend(res.cpu().numpy().tolist())
        
    def evaluate(self, UQ_ref=None, energy_ref=None, energy_ref_list=None, val_mod=False):
        # UQ_ref: the threshold selected according to validation logits, for calculating uncertainty in Bayes-CAL
        # energy_ref: the threshold selected according to validation energy, for calculating energy in CRoFT
        results = OrderedDict()
        if energy_ref is None:
            acc = 100.0 * self._correct / self._total
            err = 100.0 - acc
            macro_f1 = 100.0 * f1_score(
                self._y_true,
                self._y_pred,
                average="macro",
                labels=np.unique(self._y_true)
            )

            # The first value will be returned by trainer.test()
            results["accuracy"] = acc
            results["error_rate"] = err
            results["macro_f1"] = macro_f1
            results["UQ_c"] = np.sum(self._UQ_c)/self._correct if self._correct > 0 else 0
            results["UQ_e"] = np.sum(self._UQ_e)/(self._total - self._correct)
            results["loss_test"] = np.sum(self.loss_test)/(self._total - self._correct)

            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n"
                f"* accuracy: {acc:.2f}%\n"
                f"* error: {err:.2f}%\n"
                f"* macro_f1: {macro_f1:.2f}%"
            )
        ###################################### Uncertainty Quantity in Bayes-CAL ######################################
        UQ_theta = np.percentile(np.array(self._UQ_c), 5) if np.sum(self._UQ_c) > 0 else 0

        if UQ_ref is not None:
            print('threshold @ TPR95 on test and on val:', UQ_theta, UQ_ref)
            UQ_theta = UQ_ref

            print('results["UQ_c"]:', np.sum(self._UQ_c) / self._correct)
            UQ_e = np.sum(self._UQ_e) / (self._total - self._correct) if self._total - self._correct > 0 else 0
            print('results["UQ_e"]:', UQ_e)
            print('results["loss_test"]', np.sum(self.loss_test) / self._total)

            FPR95 = len(np.where(np.array(self._UQ_e) > UQ_theta)[0].tolist())
            max_UQ_e = np.max(self._UQ_e) if len(self._UQ_e) > 0 else 0
            print('max confidence score of ood datasets', max_UQ_e)
            max_score_id = np.max(self._UQ_c) if np.sum(self._UQ_c) > 0 else 0
            print('max confidence score of id datasets', max_score_id)
            FPR95 = FPR95/len(self._UQ_e) if len(self._UQ_e) > 0 else -1
            print('FPR95:', FPR95)
            self.detect = np.where(np.array(self.res_all) > UQ_theta, 1, 0)
            if len(self._UQ_e) > 0 and len(self._UQ_c) > 0:
                AUROC = roc_auc_score(np.array(self.confidence_gt), self.detect)
            else:
                AUROC = None
            print('AUROC:', AUROC)
            all = (self._total - len(np.where(np.array(self._UQ_e) <= UQ_theta)[0].tolist()) -
                   len(np.where(np.array(self._UQ_c) <= UQ_theta)[0].tolist()))
            if all != 0:
                print('ACCURACY:', 100.0 * (self._correct - len(np.where(np.array(self._UQ_c) <= UQ_theta)[0].tolist())) / all)
            else:
                print('ACCURACY:', acc)

        results["UQ_threshold"] = UQ_theta
        #######################################################################################################

        ###################################### Energy Estimation in CRoFT ######################################
        energy_theta = np.percentile(np.array(self._EE), 5) if len(self._EE) > 0 else -1
        if energy_ref is not None:
            print("=> ood detection result")
            print('results["EE"]:', np.sum(self._EE) / len(self._EE))
            print('threshold @ TPR95 on ID val:', energy_ref)
            print('threshold @ TPR95 on OOD test:', energy_theta)
            energy_theta = energy_ref
            OODD_FPR95 = len(np.where(np.array(self._EE) > energy_theta)[0].tolist())
            OODD_FPR95 = OODD_FPR95 / len(self._EE) if len(self._EE) > 0 else -1
            print('OODD_FPR95:', OODD_FPR95)
            if energy_ref_list is not None:
                gt = np.hstack([np.zeros(len(self._EE), dtype=np.int32), np.ones(len(energy_ref_list), dtype=np.int32)])
                AUROC = roc_auc_score(gt, np.hstack([np.array(self._EE), np.array(energy_ref_list)]))
                print('AUROC:', AUROC)
        results["energy_threshold"] = energy_theta
        if val_mod:
            results["energy_list"] = np.array(self._EE)  # not for training, just for ood detection
        #########################################################################################################
        
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%".format(
                        label, classname, total, correct, acc
                    )
                )
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
