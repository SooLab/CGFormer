import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from PIL import Image
from loguru import logger
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, (image, text, target, l_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = torch.stack(image).cuda(non_blocking=True)
        text = torch.stack(text).cuda(non_blocking=True)
        target = torch.stack(target).cuda(non_blocking=True)
        l_mask = torch.stack(l_mask).cuda(non_blocking=True)
        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear', align_corners=True)
        text = text.squeeze(1)
        l_mask = l_mask.squeeze(1)
        # forward
        with amp.autocast():
            pred, target, loss = model(image, text, l_mask, target)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # if args.max_norm:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(optimizer.param_groups[0]["lr"])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    I = []
    U = []
    model.eval()
    time.sleep(2)
    for imgs, text, masks, l_mask in val_loader:
        # data
        imgs = torch.stack(imgs).cuda(non_blocking=True)
        text = torch.stack(text).cuda(non_blocking=True)
        l_mask = torch.stack(l_mask).cuda(non_blocking=True)
        text = text.squeeze(1)
        l_mask = l_mask.squeeze(1)
        # inference
        preds, maps = model(imgs, text, l_mask)
        preds = torch.sigmoid(preds)
        # process one batch
        for pred, mask in zip(preds, masks):
            # iou
            pred = pred.cpu().numpy()
            mask = mask.cpu().numpy()
            pred = np.array(pred > 0.5)
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            I.append(np.sum(inter))
            U.append(np.sum(union))
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    I = np.stack(I)
    I = torch.from_numpy(I).to(imgs.device)
    I = concat_all_gather(I).sum()

    U = np.stack(U)
    U = torch.from_numpy(U).to(imgs.device)
    U = concat_all_gather(U).sum()
    oIoU = I/U
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  mIoU={:.2f}  oIoU={:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100.*(oIoU))
    logger.info(head + temp)
    return oIoU, prec


@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    I = 0.
    U = 0.
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for ori_img, img, texts, mask, l_masks, seg_id, sents in tbar:
        img = img.cuda(non_blocking=True)
        mask = mask.cpu().numpy()
        for text, l_mask, sent in zip(texts, l_masks, sents):
            text = text.cuda(non_blocking=True)
            l_mask = l_mask.cuda(non_blocking=True)

            text = text.squeeze(1)
            l_mask = l_mask.squeeze(1)

            # inference
            pred, maps = model(img, text, l_mask)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != ori_img.shape[:-1]:
                pred = F.interpolate(pred, size=ori_img.shape[1:-1], mode='bicubic', align_corners=True)
            # # process one sentence
            pred = pred.cpu().numpy()
            pred_ = np.array(pred > 0.5) 
            inter = np.logical_and(pred_, mask)
            union = np.logical_or(pred_, mask)
            I += np.sum(inter)
            U += np.sum(union)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
       
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('oIoU={:.2f}'.format(100.*(I/U)))
    logger.info('mIoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec
