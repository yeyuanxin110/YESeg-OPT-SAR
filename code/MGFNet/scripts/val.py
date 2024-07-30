import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from models import *
from datasets import *
from augmentations import get_val_augmentation
from metrics import Metrics
from utils.utils import setup_cudnn
import os
import numpy as np
from models.MGFNet import MGFNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    for opt,sar, labels in tqdm(dataloader):
        opt = opt.to(device)
        sar = sar.to(device)
        labels = labels.to(device)
        preds = model(opt,sar).softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    accs, macc = metrics.compute_pixel_acc()
    f1s, mf1 = metrics.compute_f1()
    oa = metrics.compute_oa()
    kappa = metrics.compute_kappa()
    recalls, mrecall = metrics.compute_recall()
    # For YESeg class-0 and 7 will be ignored
    iou = ious[1:7]
    acc = accs[1:7]
    f1  = f1s[1:7]
    recall = recalls[1:7]

    miou = np.mean(iou)
    macc = np.mean(acc)
    mf1 = np.mean(f1)
    mrecall = np.mean(recall)

    return acc, macc, f1, mf1, iou, miou, oa, recall, mrecall, kappa



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()

    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    train_cfg = cfg['TRAIN']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")
    
    if dataset_cfg['NAME'] =='YESeg_OPT_SAR':
        model = MGFNet(num_classes=dataset.n_classes,pretrained=False)
    else:
        print("Dataset not supported")

    if train_cfg['DP']:
        model = torch.nn.DataParallel(model)
    else:
        model = model

    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))

    model = model.to(device)


    acc, macc, f1, mf1, ious, miou, oa, recall, mrecall, kappa = evaluate(model, dataloader, device)

    table = {
        'Class': list(dataset.CLASSES)[1:7] + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc],
        'Recall': recall + [mrecall]
    }

    print(tabulate(table, headers='keys'))
    print(f"OA: {oa}")
    print(f"Kappa: {kappa}")
    print(f"Mean IoU: {miou}")
    print(f"Mean F1: {mf1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='E:\wk\MGFNet\configs\MGFNet_YESeg.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)