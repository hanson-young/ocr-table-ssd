from JNetV3.utils.preprocessing import *
from JNetV3.utils.train import train
from JNetV3.utils.logs import trainlog
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler,Adam,RMSprop
from JNetV3.utils.preprocessing import gen_dataloader
from JNetV3.models.losses import CrossEntropyLoss2d,MSELoss2d
from JNetV3.models.imJNetV3 import Mobile_Unet
from JNetV3.Sdata.Saug import *
from JNetV3.utils.preprocessing import gen_dataloader
import logging

class trainAug(object):
    def __init__(self, size=(768, 768)):
        self.augment = Compose([
            RandomSelect([
                RandomRotate(angles=(-30, 30), bound=None),
                RandomResizedCrop(size=size),
                RandomSmall(ratio=0.15),
            ]),
            RandomBrightness(delta=30),
            ResizeImg(size=size),
            RandomHflip(),
            RandomVflip(),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)


class valAug(object):
    def __init__(self, size=(768, 768)):
        self.augment = Compose([
            # RandomSelect([
            #     RandomRotate(angles=(-40, 40), bound=None),
            #     RandomResizedCrop(size=size),
            #     RandomSmall(ratio=0.15),
            # ]),
            # RandomBrightness(delta=30),
            ResizeImg(size=size),
            # RandomHflip(),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# img_root = '/media/hszc/data1/seg_data/'

# resume = "/home/handsome/Documents/DefectInspection/RCF-pytorch/JNetV3/model_tmp_mse/weights-553-9-[0.716].pth"

resume = None
#
start_epoch = 0

bs = 32
save_dir = '/home/handsome/Documents/code/orc-table-ssd/JNetV3/model_tmp_mse'
model = Mobile_Unet(num_classes=1,alpha=0.15,alpha_up=0.25)
train_root = '/media/handsome/backupdata/hanson/orc_cropped/segmentation/train'
val_root = '/media/handsome/backupdata/hanson/orc_cropped/segmentation/test'

# saving dir
save_dir = save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)

train_pd, _ = get_train_val(train_root, test_size=0.0)
_, val_pd = get_train_val(val_root, test_size=1.0)

data_set, data_loader = gen_dataloader(train_pd, val_pd, trainAug(), valAug(), train_bs =bs, val_bs=bs,
                                train_shuffle=True, val_shuffle=False, dis=None)
print(len(data_set['train']), len(data_set['val']))


logging.info(model)

criterion = torch.nn.MSELoss()

optimizer = RMSprop(model.parameters(), lr=1e-3, alpha=0.9)
# exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
model.cuda()
if resume:
    model.eval()
    logging.info('resuming finetune from %s' % resume)
    try:
        model.load_state_dict(torch.load(resume, map_location=lambda storage, loc: storage))
    except KeyError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(resume))
    # optimizer.load_state_dict(torch.load(os.path.join(save_dir, 'optimizer-state.pth')))


train(model,
      epoch_num=10000,
      start_epoch=start_epoch,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=None,
      data_set=data_set,
      data_loader=data_loader,
      save_dir=save_dir,
      print_inter=50,
      val_inter=500,
      )