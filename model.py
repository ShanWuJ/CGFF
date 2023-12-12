import torch.nn as nn
import torch
from generator import Generator
from swapper import Swapper

class BranchClassification(nn.Module):
    def __init__(self,outdim,classe_num):
        super(BranchClassification,self).__init__()
        self.max1 = nn.MaxPool2d(kernel_size=28,stride=28)
        self.max2 = nn.MaxPool2d(kernel_size=28,stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=28,stride=28)
        self.max4 = nn.MaxPool2d(kernel_size=28,stride=28)
        self.outsize=outdim
        self.conv5_num_fs = 2048 * 1 * 1

        self.classifier1 = BasicClass(self.conv5_num_fs//2,outdim,kernel_size=1,class_num=classe_num)
        self.classifier2 = BasicClass(self.conv5_num_fs // 2, outdim, kernel_size=1, class_num=classe_num)
        self.classifier3 = BasicClass(self.conv5_num_fs // 2, outdim, kernel_size=1, class_num=classe_num)
        self.classifier4 = BasicClass(self.conv5_num_fs // 2, outdim, kernel_size=1, class_num=classe_num)
        self.classifier_concat=BasicClass(1024 * 5,outdim,kernel_size=1,class_num=classe_num)

    def forward(self,f1,f2,f3,f4,f6):
        f1 = self.max1(f1)
        f2 = self.max2(f2)
        f3 = self.max3(f3)
        f4 = self.max4(f4)
        f_concat=torch.cat((f1,f2,f3,f4,f6),1)

        cls1=self.classifier1(f1)
        cls2=self.classifier2(f2)
        cls3=self.classifier3(f3)
        cls4=self.classifier4(f4)
        cls_concat=self.classifier_concat(f_concat)
        return cls1, cls2, cls3,cls4,cls_concat

class MGS(nn.Module):
    def __init__(self, model,outdim=512,classes_num=200):
        super(MGS, self).__init__()
        self.features = model
        self.outsize=outdim
        self.max6 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.elu = nn.ELU(inplace=True)
        self.conv5_num_fs = 2048 * 1 * 1
        self.branch1 = BranchClassification(outdim=outdim,classe_num=classes_num)
        self.branch2 = BranchClassification(outdim=outdim,classe_num=classes_num)

        self.basic_block1= nn.Sequential(
            BasicConv(self.conv5_num_fs//2, outdim, kernel_size=1, stride=1, padding=0, elu=True),
            BasicConv(outdim, self.conv5_num_fs // 2, kernel_size=3, stride=1, padding=1, elu=True),
        )
        self.basic_block2 = nn.Sequential(
            BasicConv(self.conv5_num_fs // 2, outdim, kernel_size=1, stride=1, padding=0, elu=True),
            BasicConv(outdim, self.conv5_num_fs // 2, kernel_size=3, stride=1, padding=1, elu=True),
        )
        self.basic_block3 = nn.Sequential(
            BasicConv(self.conv5_num_fs, outdim, kernel_size=1, stride=1, padding=0, elu=True),
            BasicConv(outdim, self.conv5_num_fs // 2, kernel_size=3, stride=1, padding=1, elu=True),
        )
        self.classifier6=BasicClass(self.conv5_num_fs // 2,outdim,kernel_size=1,class_num=classes_num,stride=1)
    # cgff
    def cgff(self,input,traning):
        if traning:
            multi_generator = Generator(input)
            granu1_map = input
            granu2_map = multi_generator.sub_generator(2)
            granu3_map = multi_generator.sub_generator(4)
            granu4_map = multi_generator.sub_generator(7)

            granu_swap = Swapper()
            swap12,swap21 = granu_swap.swap(granu1_map,granu2_map)
            swap213, swap32 = granu_swap.swap(swap21, granu3_map)
            swap324, swap43 = granu_swap.swap(swap32, granu4_map)
            swap431, swap14 = granu_swap.swap(swap43, granu1_map)
            swap142, _ = granu_swap.swap(swap14,granu2_map)

            f1=swap142
            f2=swap213
            f3=swap324
            f4=swap431
        else:
            f1=input
            f2=input
            f3=input
            f4=input
        return f1,f2,f3,f4

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)
        xl1 = self.basic_block1(xf4)
        xl2 = self.basic_block2(xf4)
        xl3 = self.basic_block3(xf5)
        training=self.training
        f11,f12,f13,f14=self.cgff(xl1,training)
        f21,f22,f23,f24=self.cgff(xl2,training)
        f6=xl3
        f6 = self.max6(f6)
        cls11, cls12, cls13, cls14, cls_cat1 = self.branch1(f11,f12,f13,f14,f6)
        cls21, cls22, cls23, cls24, cls_cat2 = self.branch2(f21,f22,f23,f24,f6)
        cls1 = [cls11, cls12, cls13, cls14, cls_cat1]
        cls2 = [cls21, cls22, cls23, cls24, cls_cat2]
        cls6 = self.classifier6(f6)
        return cls1,cls2,cls6


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, elu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.elu = nn.ELU() if elu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.elu is not None:
            x = self.elu(x)
        return x

class BasicClass(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,class_num,stride=1):
        super(BasicClass,self).__init__()
        self.bn1=nn.BatchNorm2d(in_channel)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size,stride)
        self.elu=nn.ELU(inplace=True)
        self.fc=nn.Linear(out_channel,class_num)

    def forward(self,x):
        out=self.bn1(x)
        out=self.conv(out)
        out=self.bn2(out)
        out=self.elu(out)
        out=out.view(out.size(0),-1)
        cls=self.fc(out)
        return cls

