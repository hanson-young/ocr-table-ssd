import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.utils import model_zoo
from torchvision import models


class imJNetV3(nn.Module):

    def __init__(self, num_classes,alpha=0.15, alpha_up=0.25):
        super(imJNetV3, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, int(oup), 1, 1, 0, bias=False),
                nn.BatchNorm2d(int(oup)),
                nn.ReLU(inplace=True),
            )
        self.b001 = nn.Sequential(
            conv_bn(3,  int(32 * alpha), 1),
        )

        self.b002 = nn.Sequential(
            conv_bn(int(32 * alpha), int(32 * alpha), 1),
        )

        self.b00 = nn.Sequential(
            conv_bn(int(32*alpha),  int(32*alpha), 2),
        )

        self.b01 = nn.Sequential(
            conv_dw(int(32*alpha),  int(64*alpha), 1),)

        self.b03 = nn.Sequential(
            conv_dw(int(64*alpha), int(96*alpha), 2),
            conv_dw(int(96*alpha), int(96*alpha), 1),)

        self.b05 = nn.Sequential(
            conv_dw(int(96*alpha), int(192*alpha), 2),
            conv_dw(int(192*alpha), int(192*alpha), 1),)

        self.b11 = nn.Sequential(
            conv_dw(int(192*alpha), int(384*alpha), 2),
            conv_dw(int(384*alpha), int(384*alpha), 1),
            conv_dw(int(384*alpha), int(384*alpha), 1),
            conv_dw(int(384*alpha), int(384*alpha), 1),
            conv_dw(int(384*alpha), int(384*alpha), 1),
            conv_dw(int(384*alpha), int(384*alpha), 1),
        )

        self.b13 = nn.Sequential(
            conv_dw(int(384*alpha), int(768*alpha), 2),
            conv_dw(int(768*alpha), int(768*alpha), 1),
        )

        self.ConvTranspose1= nn.ConvTranspose2d(int(768*alpha), int(384*alpha)+1, 2, stride=2)
        self.b14 = conv_dw(int(768*alpha), int(384*alpha*alpha_up), 1)

        self.ConvTranspose2= nn.ConvTranspose2d(int(384*alpha*alpha_up), int(192*alpha), 2, stride=2)
        self.b15 = conv_dw(int(384*alpha)-1, int(192*alpha*alpha_up), 1)

        self.ConvTranspose3= nn.ConvTranspose2d(int(192*alpha*alpha_up), int(96*alpha), 2, stride=2)
        self.b16 = conv_dw(int(192*alpha), int(96*alpha*alpha_up), 1)

        self.ConvTranspose4= nn.ConvTranspose2d(int(96*alpha*alpha_up), int(64*alpha), 2, stride=2)
        self.b17 = conv_dw(int(96*alpha)+4, int(64*alpha*alpha_up), 1)

        self.b18 = conv_bn( int(64*alpha*alpha_up)+int(32*alpha), int(32*alpha*alpha_up)+1, 1)

        self.final_cat = nn.Conv2d(int(32 * alpha * alpha_up) + 1, int(32 * alpha), 1)
        self.final = nn.Conv2d(int(32 * alpha) * 2, num_classes, 1)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        b001 =self.b001(x)
        b002 = self.b002(b001)
        b00 = self.b00(b002)
        # print b00.size()
        b01 =self.b01(b00)
        # print b01.size()

        b03 =self.b03(b01)
        # print b03.size()

        b05 =self.b05(b03)
        # print b05.size()
        b11 =self.b11(b05)
        # print b11.size()
        b13 =self.b13(b11)
        # print b13.size()

        up1 = torch.cat([self.ConvTranspose1(b13),b11],1)
        # print up1.size()
        b14 = self.b14(up1)
        # print b14.size()

        up2 = torch.cat([self.ConvTranspose2(b14),b05],1)
        # print up2.size()
        b15 = self.b15(up2)
        # print b15.size()

        up3 = torch.cat([self.ConvTranspose3(b15),b03],1)
        # print up3.size()
        b16 = self.b16(up3)
        # print b16.size()
        up4 = torch.cat([self.ConvTranspose4(b16),b01],1)
        # print up4.size()
        b17 = self.b17(up4)

        # print b17.size()
        up5 = torch.cat([b17,b00],1)
        # print up5.size()
        b18=self.b18(up5)
        # print b18.size()
        b19 = F.upsample_bilinear(self.final_cat(b18),scale_factor=2)
        b20 = torch.cat([b19, b002], 1)
        b21 = self.final(b20)
        return b21, x
if __name__=='__main__':
    from torch.autograd import Variable
    import numpy as np
    x = torch.FloatTensor(1,3,768,768)
    x = Variable(x)
    model = imJNetV3(num_classes=1, alpha=0.15)
    print(model)
    y = model(x)
    seg_mask = torch.squeeze(y)
    seg_mask = seg_mask.detach().cpu().numpy().astype(np.float32)
    print(y.view(-1,768,768).size())
