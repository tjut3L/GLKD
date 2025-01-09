from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F

from .util import AverageMeter, accuracy, reduce_tensor
from models.util import Embed, ConvReg, LinearEmbed, SimKD, SelfA
from distiller_zoo import DKD, KD, DIST, NKD, Attention, RKDLoss, HintLoss, NKDLoss


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # input = input.float()
        input = [im.float() for im in input]
        if torch.cuda.is_available():
            input = [im.cuda() for im in input]
            # input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input[0])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input[0].size(0))
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))

        # output = model(input)
        # loss = criterion(output, target)
        #
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()

    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    if opt.distill == 'ctkd':
        model_st = module_list[1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = [im.float() for im in input]
        if torch.cuda.is_available():
            input = [im.cuda() for im in input]
            # input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        # if opt.distill == 'tts' or opt.distill == 'simkd' or opt.distill == 'tts_dkd' or opt.distill == 'tts_dist':
        if opt.distill in ['tts', 'simkd', 'simkd_dkd', 'tts_dkd', 'tts_dist', 'ctkd1']:
            preact = False
            feat_s, logit_s = model_s(input[0], is_feat=True, preact=preact)
            with torch.no_grad():
                feat_tg, logit_tg = model_t(input[0], is_feat=True, preact=preact)
                feat_tl, logit_tl = model_t(input[1], is_feat=True, preact=preact)
                # feat_tl, logit_tl = model_t(input[1], is_feat=True, preact=preact)
                feat_tg = [f.detach() for f in feat_tg]
                feat_tl = [f.detach() for f in feat_tl]

                # logit_t = logit_tg * 0.8 + logit_tl * 0.2
                # logit_t = logit_tg * (opt.tg) + logit_tl * (opt.tl)
        elif opt.distill == 'ctkd':
            feat_s, logit_s = model_s(input[0], is_feat=True)
            feat_st, logit_st = model_st(input[0], is_feat=True)
            with torch.no_grad():
                feat_t, logit_t = model_t(input[1], is_feat=True)
                feat_t = [f.detach() for f in feat_t]


        else:
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, logit_s = model_s(input[0], is_feat=True, preact=preact)

            with torch.no_grad():
                feat_t, logit_t = model_t(input[0], is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

        cls_t = model_t.get_feat_modules()[-1]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)

        if opt.distill == "tts":
            loss_div = criterion_kd(logit_s, logit_tg) * opt.tg + criterion_kd(logit_s, logit_tl) * opt.tl
        else:
            # loss_div = criterion_div(logit_s, logit_t)
            loss_div = 0

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_div = criterion_div(logit_s, logit_t)
            loss_kd = 0
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_tg[-2], cls_t)
            logit_s = pred_feat_s
            criterion_div = DKD()
            loss_tg = criterion_div(logit_s, logit_tg, target, epoch)
            loss_tl = criterion_div(logit_s, logit_tl, target, epoch)
            loss_div = loss_tg * opt.tg + loss_tl * opt.tl

            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        elif opt.distill == 'simkd_dkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_tg[-2], cls_t)
            logit_s = pred_feat_s
            criterion_div = DKD()
            loss_tl = criterion_div(logit_s, logit_tl, target, epoch)
            loss_div = loss_tl * opt.tl
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)

        elif opt.distill == 'ctkd0':
            if opt.method == 'dkd':
                criterion_kd = DKD()
                loss_kd = criterion_kd(logit_s, logit_t, target, epoch)  # L kd
            elif opt.method == 'attention':
                criterion_kd = Attention()
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.method == 'fitnet':
                f_s = module_list[2](feat_s[opt.hint_layer])
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.method == 'mse':
                loss_kd = F.mse_loss(logit_s, logit_t)
            elif opt.method == 'l1':
                loss_kd = F.l1_loss(logit_s, logit_t)
            elif opt.method == 'dist':
                criterion_kd = DIST()
                loss_kd = criterion_kd(logit_s, logit_t)
            elif opt.method == 'rkd':
                criterion_kd = RKDLoss()
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.method == 'nkd':
                criterion_kd = NKDLoss()
                loss_kd = criterion_kd(logit_s, logit_t, target)
            elif opt.method == 'srrl':
                trans_feat_s, pred_feat_s = module_list[2](feat_s[-1], cls_t)
                loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
            elif opt.method == 'cos':
                loss_kd = F.cosine_embedding_loss(logit_s, logit_t, target)

            loss_cls = criterion_cls(logit_s, target)  # L cls student

            y_tech_temp = torch.autograd.Variable(logit_st.data, requires_grad=False)  # L2
            # loss_course = ((y_tech_temp - logit_s) * (y_tech_temp - logit_s)).sum() / input[0].shape[0]  # L2
            loss_st = F.mse_loss(y_tech_temp, logit_s)
            loss_t = criterion_cls(logit_st, target)

        elif opt.distill == 'ctkd':
            criterion_kd = DKD()
            loss_cls = criterion_cls(logit_s, target)  # L cls student
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)  # L kd
            loss_t = criterion_cls(logit_st, target)
            y_tech_temp = torch.autograd.Variable(logit_st.data, requires_grad=False)  # L2
            # loss_course = ((y_tech_temp - logit_s) * (y_tech_temp - logit_s)).sum() / input[0].shape[0]  # L2
            if opt.method == 'mse':
                loss_st = F.mse_loss(y_tech_temp, logit_s)
            elif opt.method == 'l1':
                loss_st = F.l1_loss(y_tech_temp, logit_s)
            elif opt.method == 'cos':
                loss_st = F.cosine_embedding_loss(logit_s, y_tech_temp, target)
            elif opt.method == 'kl':
                loss_st = criterion_div(y_tech_temp, logit_s)
        elif opt.distill == 'ctkd1':
            criterion_kd = DKD()
            loss_cls = criterion_cls(logit_s, target)  # L cls student
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)  # L kd

            y_tech_temp = torch.autograd.Variable(logit_st.data, requires_grad=False)  # L2
            # loss_course = ((y_tech_temp - logit_s) * (y_tech_temp - logit_s)).sum() / input[0].shape[0]  # L2
            loss_st = F.mse_loss(y_tech_temp, logit_s)
            loss_t = criterion_cls(logit_st, target)  # L cls teacher

            # g_s = feat_s[1:-1]
            # g_t = feat_t[1:-1]
            # loss_group = criterion_kd(g_s, g_t)
            # loss_at = sum(loss_group)  # L at

        elif opt.distill == 'ctkd1':
            criterion_div = DKD()
            loss_div = criterion_div(logit_s, logit_tg, target, epoch)
            loss_kd = F.mse_loss(logit_s, logit_tl)

        elif opt.distill == 'tts':
            loss_kd = 0
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch)
        elif opt.distill == 'nkd':
            loss_kd = criterion_kd(logit_s, logit_t, target)
        elif opt.distill == 'tts_dkd':
            loss_div = criterion_kd(logit_s, logit_tg, target, epoch) * opt.tg + criterion_kd(logit_s, logit_tl, target,
                                                                                              epoch) * opt.tl
            loss_kd = 0
        elif opt.distill == 'tts_dist':
            loss_div = criterion_kd(logit_s, logit_tg) * opt.tg + criterion_kd(logit_s, logit_tl) * opt.tl
            loss_kd = 0
        elif opt.distill == 'tts_rdist':
            f_s = feat_s[-1]
            f_tg = feat_tg[-1]
            f_tl = feat_tl[-1]
            loss_div = criterion_kd(logit_s, logit_tg, f_s, f_tg) * opt.tg + criterion_kd(logit_s, logit_tl, f_s,
                                                                                          f_tl) * opt.tl
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill in ['ctkd']:
            loss = opt.alpha * loss_cls + opt.beta * loss_kd + opt.gamma * loss_st + opt.t * loss_t

        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input[0].size(0))
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill in ['simkd', 'simkd_dist']:
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
                    model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_st(val_loader, model, criterion, opt):
    """st_validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('st_Test: [{0}/{1}]\t'
                      'st_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'st_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'st_Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'st_Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * st_Acc@1 {top1.avg:.3f} st_Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
