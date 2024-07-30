import sys
import os
from os.path import dirname as up
import torch
import argparse
import yaml
import time
import numpy as np
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from models import *
from datasets import *
from augmentations import get_train_augmentation, get_val_augmentation
from losses import get_loss
from schedulers import get_scheduler
from optimizers import get_optimizer
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate
from models.MGFNet import MGFNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

def main(cfg, gpu, save_dir):

    start = time.time()

    best_mIoU = 0.0

    # device
    device = torch.device(cfg['DEVICE'])
    
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    num_workers = train_cfg['NUM_WORKERS']
    
    # dataset
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform)

    if dataset_cfg['NAME'] =='YESeg_OPT_SAR':
        model = MGFNet(num_classes=trainset.n_classes,pretrained=True)
    else:
        print("Dataset not supported")

    model = model.to(device)

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    elif train_cfg['DP']:
        sampler = RandomSampler(trainset)
        model = torch.nn.DataParallel(model)
    else:
        sampler = RandomSampler(trainset)
        model = model
    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']

    weights = np.array([0,1,1,1,1,1,1,0], np.float32)
    class_weights = torch.from_numpy(weights).to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, class_weights)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (opt,sar, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            opt = opt.to(device)
            sar = opt.to(device)
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):

                logits = model(opt,sar)
                loss_oce = loss_fn[0](logits, lbl) #ce
                loss_focal = loss_fn[1](logits, lbl) #focal
                loss_dice = loss_fn[2](logits, lbl) #dice

                loss = loss_oce+loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            acc, macc, f1, mf1, ious, miou, oa, recall, mrecall, kappa= evaluate(model, valloader, device)
            writer.add_scalar('val/mIoU', miou, epoch)
            writer.add_scalar('val/mAcc', macc, epoch)
            writer.add_scalar('val/mF1', mf1, epoch)
            writer.add_scalar('val/OA', oa, epoch)
            writer.add_scalar('val/Kappa', kappa, epoch)
            writer.add_scalar('val/mrecall', mrecall, epoch)

            print(f"Current OA: {oa:.3f},Kappa: {kappa:.3f}, mIoU: {miou:.3f}, mAcc: {macc:.3f}, mF1: {mf1:.3f}, mrecall: {mrecall:.3f}")
            table = {
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc],
                'Recall': recall + [mrecall]
            }
    
            print(tabulate(table, headers='keys'))
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current IoU: {miou} Best IoU: {best_mIoU}")
            if (epoch+1) == epochs:
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}+_final.pth")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best acc', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='E:\wk\MGFNet\configs\MGFNet_YESeg.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407) # Set seed for reproducibility
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    os.makedirs(save_dir, exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()