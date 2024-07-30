import torch
from torch import Tensor
from typing import Tuple
from sklearn.metrics import cohen_kappa_score


class Metrics:
    def __init__(self, num_classes: int, ignore_label, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        if self.ignore_label is not None:
            keep = target != self.ignore_label
            target = target[keep]
            pred = pred[keep]
        self.hist += torch.bincount(target * self.num_classes + pred, minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def compute_oa(self) -> float:
        # For YESeg class-0 and 7 will be ignored
        valid_hist = self.hist[1:-1, 1:-1]
        oa = valid_hist.diag().sum() / valid_hist.sum()
        # oa = self.hist.diag().sum() / self.hist.sum()
        return round(oa.item() * 100, 2)

    def compute_jaccard(self) -> float:
        ious = self.compute_iou()[0]
        jaccard = sum(ious) / len(ious)
        return round(jaccard, 2)

    def compute_kappa(self) -> float:

        valid_hist = self.hist[1:-1, 1:-1]
        # valid_hist = self.hist

        n = valid_hist.sum().item()

        diag_sum = valid_hist.diag().sum().item()

        row_sum = valid_hist.sum(dim=1).float()
        col_sum = valid_hist.sum(dim=0).float()

        expected_diag_sum = (row_sum * col_sum).sum().item() / n

        kappa = (diag_sum - expected_diag_sum) / (n - expected_diag_sum)

        return round(kappa * 100, 2)

    def compute_recall(self) -> Tuple[Tensor, Tensor]:
        recall = self.hist.diag() / self.hist.sum(0)
        mrecall = recall[~recall.isnan()].mean().item()
        recall *= 100
        mrecall *= 100
        return recall.cpu().numpy().round(2).tolist(), round(mrecall, 2)
    
    def get_confusion_matrix_tensor(self) -> Tensor:
        return self.hist.cpu()