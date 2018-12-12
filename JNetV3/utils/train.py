from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from math import ceil
from copy import deepcopy
# from JNetV3.logs import *
from JNetV3.utils.preprocessing import *
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from JNetV3.utils.plotting import getPlotImg
from JNetV3.utils.predicting import predict
from JNetV3.utils.metrics import metrics_pred, precision, recall, f1_score

dt = datetime.datetime.now().strftime('%b-%d-%h-%m-%s')
def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    # num_positive = torch.sum((mask==1).float()).float()
    # num_negative = torch.sum((mask==0).float()).float()
    #
    # mask[mask == 1.0] = 1.0 * num_negative / (num_positive + num_negative)
    # mask[mask == 0.0] = 1.1 * num_positive / (num_positive + num_negative)

    mask[mask == 1.0] = 0.8
    mask[mask == 0.0] = 0.1

    # mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

# https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def weighted_mse_loss(prediction, label):
    label = label.long()
    weights = label.float()

    # weights[weights == 1.0] = 8
    # weights[weights == 0.0] = 2
    weights[weights == 1.0] = 1
    weights[weights == 0.0] = 1
    pct_var = (prediction.float()-label.float())**2
    out = pct_var * weights.expand_as(label)
    loss = out.mean()
    return loss

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500,
          ):
    writer = SummaryWriter(save_dir)
    best_model_wts = model.state_dict()
    best_f1 = 0
    val_loss = 0
    train_loss = 0
    # running_loss = 20
    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        # exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode


        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            if step % val_inter == 0:
                # val phase
                model.eval()
                # loss_fn = weighted_mse_loss
                # loss_fn = cross_entropy_loss_RCF
                loss_fn = criterion
                t0 = time.time()

                test_precisions, test_recalls, test_f1_scores, val_loss = predict(loss_fn, model, data_set['val'], data_loader['val'], counting=False)

                t1 = time.time()
                since = t1 - t0

                logging.info('--' * 30)
                # logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d] | val_loss: %.4f | precisions: %.4f | recalls: %.4f | f1_scores: %.4f | time: %d'
                             % (dt, epoch, val_loss, test_precisions, test_recalls, test_f1_scores,  since))

                if test_f1_scores > best_f1:
                    best_f1 = test_f1_scores
                    best_model_wts = deepcopy(model.state_dict())

                # save model
                save_path1 = os.path.join(save_dir,
                                          'weights-%d-%d-[%.3f].pth' % (epoch, batch_cnt, test_f1_scores))
                torch.save(model.state_dict(), save_path1)
                save_path2 = os.path.join(save_dir,
                                          'optimizer-state.pth')
                torch.save(optimizer.state_dict(), save_path2)

                logging.info('saved model to %s' % (save_path1))
                logging.info('--' * 30)


            model.train(True)
            imgs, masks, _ = data

            imgs = Variable(imgs.cuda())
            masks = Variable(masks.cuda(),requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(imgs)

            outputs = outputs.view(-1, outputs.size()[2], outputs.size()[3])
            # print outputs.size(), masks.size()
            if outputs.size() != masks.size():
                outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')

            # print outputs.size()
            # print masks.size()
            mask_loss = criterion(outputs, masks)
            # mask_loss = weighted_mse_loss(outputs, masks)
            # mask_loss = cross_entropy_loss_RCF(outputs, masks)
            # mask_loss = weighted_mse_loss(outputs, masks)
            # loss = F.mean_absolute_error(outputs, masks)
            ###############################################cross entropy loss
            train_loss = mask_loss
            ###############################################
            train_loss.backward()
            optimizer.step()
            # running_loss = running_loss*0.95 + 0.05*loss.data[0]
            # running_loss = loss.data[0]

            # cal pixel acc
            # _, preds = torch.max(outputs,1)  # (bs, H, W)
            # # preds = F.softmax(outputs,dim=1).round()[:, 1, :].long()
            # batch_corrects = torch.sum((preds==masks).long()).data[0]
            # batch_acc = 1.*batch_corrects / (masks.size(0)*masks.size(1)*masks.size(2))

            true_positives, predicted_positives, possible_positives, union_areas = metrics_pred(outputs.data.cpu().numpy(),\
                             imgs.cpu().data.numpy(), masks.cpu().data.numpy())

            train_precisions = precision(true_positives, predicted_positives)
            train_recalls = recall(true_positives, possible_positives)
            train_f1_scores = f1_score(train_recalls, train_precisions)

            if step % print_inter == 0:
                logging.info('%s [%d-%d] | train_loss: %.4f | precisions: %.4f | recalls: %.4f | f1_scores: %.4f'
                             % (dt, epoch, batch_cnt, train_loss, train_precisions, train_recalls, train_f1_scores))

            # plot image
            if step % (print_inter) == 0:
                smp_img = imgs[0]  # (3, H, W)
                true_hm = masks[0]  #(H,W)
                pred_hm = outputs[0]

                imgs_to_plot = getPlotImg(smp_img, pred_hm, true_hm)

                # for TensorBoard
                imgs_to_plot = torch.from_numpy(imgs_to_plot.transpose((0,3,1,2))/255.0)
                grid_image = make_grid(imgs_to_plot, 2)
                writer.add_image('plotting',grid_image, step)
                writer.add_scalar('train_loss', train_loss , step)
                writer.add_scalar('val_loss', val_loss, step)


    # save best model
    save_path = os.path.join(save_dir,
                             'bestweights-[%.3f].pth' % (best_f1))
    torch.save(best_model_wts, save_path)
    logging.info('saved model to %s' % (save_path))

    return best_f1, best_model_wts