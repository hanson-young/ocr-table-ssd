from utils.preprocessing import *
from utils.train_rcf import train, trainlog
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler,Adam,RMSprop
from utils.preprocessing import gen_dataloader
from models.losses import CrossEntropyLoss2d,MSELoss2d

from Sdata.Saug import *
from utils.preprocessing import gen_dataloader
import logging

class trainAug(object):
    def __init__(self, size=(384, 384)):
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
    def __init__(self, size=(384, 384)):
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

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# img_root = '/media/hszc/data1/seg_data/'

# resume = "/home/handsome/Documents/DefectInspection/RCF-pytorch/JNetV3/model_tmp_mse/weights-2500-0-[0.692].pth"
#
resume = None
#
start_epoch = 0
# from models.Mobile_Unet_lite import Mobile_Unet
#
# bs = 192
# save_dir = '/home/handsome/Documents/heils_git/face-antispoofing-m/deepface-anti-spoofing/liveness_pytorch/tmp'
# model = Mobile_Unet(num_classes=1,alpha=0.15,alpha_up=0.25)
# train_root = '/home/handsome/Documents/DataSet/nir-merge-v2/train'
# val_root = '/home/handsome/Documents/DataSet/nir-merge-v2/test'
#
jsig = 1
from models.RCFNet import RCF
bs = 2
save_dir = '/home/handsome/Documents/DefectInspection/RCF-pytorch/JNetV3/model_tmp_cel'
model = RCF()
# model.cuda()
model.apply(weights_init)

train_root = '/media/handsome/data3/GlassesDefect/Bridge_Crack/CrackForest-dataset/train'

# saving dir
save_dir = save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)

train_pd, val_pd = get_train_val(train_root, test_size=0.2)
# _, val_pd = get_train_val(val_root, test_size=1.0)

data_set, data_loader = gen_dataloader(train_pd, val_pd, trainAug(), valAug(), train_bs =bs, val_bs=bs,
                                train_shuffle=True, val_shuffle=False, dis=None)
print(len(data_set['train']), len(data_set['val']))


logging.info(model)
# criterion = CrossEntropyLoss2d()

criterion = torch.nn.MSELoss()

# # learning scheduler
# step1_bs_rate = 25. / 24.
# step2_bs_rate = 50. / 24.
# step3_bs_rate = 75. / 24.
# steps_bs_rate = 100. / 24.
# step1 = int(bs * step1_bs_rate)
# step2 = int(bs * step2_bs_rate)
# step3 = int(bs * step3_bs_rate)
# steps = int(bs * steps_bs_rate)
# logging.info('lr steps1: %d' % step1)
# logging.info('lr steps2: %d' % step2)
# logging.info('lr steps3: %d' % step3)
# logging.info('total steps: %d' % steps)

# def lr_lambda(epoch):
#     if epoch < 50:
#         return 1
#     elif epoch < 75:
#         return 0.1
#     elif epoch < 90:
#         return 0.05
#     else:
#         return 0.01
# optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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