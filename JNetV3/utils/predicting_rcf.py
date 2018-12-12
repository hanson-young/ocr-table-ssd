import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from metrics import metrics_pred, precision, recall, f1_score

def predict(loss_fn, model, data_set, data_loader, counting=False):
    """ Validate after training an epoch
    Note:
    """
    model.eval()

    true_positives = []
    predicted_positives = []
    possible_positives = []
    union_areas = []
    loss = []
    for bc_cnt, bc_data in enumerate(data_loader):
        if counting:
            print('%d/%d' % (bc_cnt, len(data_set)//data_loader.batch_size))
        imgs, masks, _ = bc_data
        imgs = Variable(imgs).cuda()
        masks = Variable(masks).cuda()
        # labels = Variable(labels).cuda()

        outputs = model(imgs)

        # outputs = outputs.view(-1, outputs.size()[2], outputs.size()[3])

        # print outputs.size(), masks.size()
        # if outputs.size() != masks.size():
        #     outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')
        mask_loss = torch.zeros(1).cuda()
        for o in outputs:
            o = o.view(-1, o.size()[2], o.size()[3])

            mask_loss = mask_loss + float(loss_fn(o, masks))

        # mask_loss = mask_loss
        # loss = criterion(outputs, masks)

        loss.append(mask_loss)
        # loss.append(loss_fn(outputs, masks))
        # outputs = F.softmax(model(imgs), dim=1)
        # if outputs.size() != masks.size():
        #     outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')
        #
        # _, outputs = torch.max(outputs, dim=1)
        output = outputs[-1]
        output = output.view(-1, output.size()[2], output.size()[3])

        output = output.cpu().data.numpy()
        # labels = labels.cpu().data.numpy()
        masks = masks.cpu().data.numpy()
        imgs = imgs.cpu().data.numpy()

        true_positive, predicted_positive, possible_positive, union_area = metrics_pred(output,imgs,masks)

        true_positives += true_positive
        predicted_positives += predicted_positive
        possible_positives += possible_positive
        union_areas += union_area
    precisions = precision(true_positives, predicted_positives)
    recalls = recall(true_positives, possible_positives)
    f1_scores = f1_score(recalls, precisions)
    loss = torch.tensor(loss)
    return precisions, recalls, f1_scores, loss.mean()
