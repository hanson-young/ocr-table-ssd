from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from math import ceil
from copy import deepcopy
from logs import *
from utils.preprocessing import *
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.plotting import getPlotImg
from utils.predicting import predict


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
    best_mIOU = 0

    running_loss = 9999
    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode


        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            model.train(True)
            imgs, masks = data

            imgs = Variable(imgs.cuda())
            masks = Variable(masks.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(imgs)

            # print outputs.size(), masks.size()
            loss = 0.
            for output in outputs:
                if output.size() != masks.size():
                    output = F.upsample(output, size=masks.size()[-2:], mode='bilinear')

                loss += criterion(output, masks)

            loss.backward()
            optimizer.step()
            running_loss = running_loss*0.95 + 0.05*loss.data[0]

            if step % print_inter == 0:
                logging.info('%s [%d-%d] | batch-loss: %.3f | running-loss: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data[0], running_loss))

            # plot image
            if step % (2*print_inter) == 0:
                model.eval()
                smp_img = imgs[0]  # (3, H, W)
                true_hm = masks[0]  #(H,W)
                output = outputs[2]
                if output.size() != masks.size():
                    output = F.upsample(output, size=masks.size()[-2:], mode='bilinear')

                pred_hm = F.softmax(output[0])[1]

                imgs_to_plot = getPlotImg(smp_img, pred_hm, true_hm)

                # for TensorBoard
                imgs_to_plot = torch.from_numpy(imgs_to_plot.transpose((0,3,1,2))/255.0)
                grid_image = make_grid(imgs_to_plot, 2)
                writer.add_image('plotting',grid_image, step)
                writer.add_scalar('loss', loss.data[0],step)

            if step % val_inter == 0:
                # val phase
                model.eval()

                t0 = time.time()
                mIOU = predict(model, data_set['val'], data_loader['val'], counting=True)
                t1 = time.time()
                since = t1-t0

                logging.info('--' * 30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d] | val-mIOU@1: %.3f%% | time: %d'
                             % (dt(), epoch, mIOU, since))

                if mIOU > best_mIOU:
                    best_mIOU = mIOU
                    best_model_wts = deepcopy(model.state_dict())

                # save model
                save_path1 = os.path.join(save_dir,
                        'weights-%d-%d-[%.3f].pth'%(epoch,batch_cnt,mIOU))
                torch.save(model.state_dict(), save_path1)
                save_path2 = os.path.join(save_dir,
                        'optimizer-state.pth')
                torch.save(optimizer.state_dict(), save_path2)

                logging.info('saved model to %s' % (save_path1))
                logging.info('--' * 30)

    # save last model
    save_path1 = os.path.join(save_dir,
                              'weights-%d-%d-[%.3f].pth' % (epoch, batch_cnt, mIOU))
    torch.save(model.state_dict(), save_path1)
    save_path2 = os.path.join(save_dir,
                              'optimizer-state.pth')
    torch.save(optimizer.state_dict(), save_path2)

    logging.info('saved model to %s' % (save_path1))
    logging.info('--' * 30)

    # save best model
    save_path = os.path.join(save_dir,
                             'bestweights-[%.3f].pth' % (best_mIOU))
    torch.save(best_model_wts, save_path)
    logging.info('saved model to %s' % (save_path))

    return best_mIOU, best_model_wts