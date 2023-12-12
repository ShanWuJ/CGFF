import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
import os

def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    last_combine_loss = 0
    branch1_concat_correct = 0
    branch1_combine_correct = 0
    branch2_concat_correct = 0
    branch2_combine_correct = 0
    last_combine_correct = 0
    total = 0
    idx = 0
    branch1_concat_test_Acc = 0
    branch1_combine_test_Acc = 0
    branch2_concat_test_Acc = 0
    branch2_combine_test_Acc = 0
    last_combine_Acc = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root='./Cub/test'
    testset = torchvision.datasets.ImageFolder(root=root,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            output1,output2,output6 = net(inputs)
            output11,output12,output13,output14,output1_concat=output1
            output21, output22, output23, output24,output2_concat = output2
            branch1_combine_output=sum(output1) + output6
            branch2_combine_output=sum(output2) + output6
            last_outputs_combine = branch1_combine_output + branch2_combine_output

            branch1_concat_loss = criterion(output1_concat, targets)
            branch2_concat_loss = criterion(output2_concat,targets)
            branch1_combine_loss = criterion(output11,targets)+criterion(output12,targets)+criterion(output13,targets)\
                                   +criterion(output14,targets)+criterion(output6,targets)+branch1_concat_loss
            branch2_combine_loss = criterion(output21,targets)+criterion(output22,targets)+criterion(output23,targets)+criterion(output24,targets)\
                    +criterion(output6,targets)+branch2_concat_loss
            last_combine_loss += branch1_combine_loss.item() + branch2_combine_loss.item()
            _, branch1_concat_predicted = torch.max(output1_concat.data, 1)
            _, branch1_combine_predicted = torch.max(branch1_combine_output.data, 1)
            _, branch2_concat_predicted = torch.max(output2_concat.data, 1)
            _, branch2_combine_predicted = torch.max(branch2_combine_output.data, 1)
            _, last_combine_predicted = torch.max(last_outputs_combine.data, 1)
            total += targets.size(0)

            branch1_concat_correct += branch1_concat_predicted.eq(targets.data).cpu().sum()
            branch1_combine_correct += branch1_combine_predicted.eq(targets.data).cpu().sum()
            branch2_concat_correct += branch2_concat_predicted.eq(targets.data).cpu().sum()
            branch2_combine_correct += branch2_combine_predicted.eq(targets.data).cpu().sum()
            last_combine_correct += last_combine_predicted.eq(targets.data).cpu().sum()
            #准确度
            branch1_concat_test_Acc = 100. * float(branch1_concat_correct) / total
            branch1_combine_test_Acc = 100. * float(branch1_combine_correct) / total
            branch2_concat_test_Acc = 100. * float(branch2_concat_correct) / total
            branch2_combine_test_Acc = 100. * float(branch2_combine_correct) / total
            last_combine_Acc = 100. * float(last_combine_correct) / total
            if batch_idx % 50 == 0:
                print('Step: %d | branch1_concat_test_Acc: %.3f | branch1_combine_test_Acc: %.3f '
                      '| branch2_concat_test_Acc: %.3f | branch2_combine_test_Acc:%.3f | last_combine_Acc:%.3f | branch1_loss:%.3f |'
                      ' branch2_loss:%.3f | last_loss:%.3f' % (
                batch_idx, branch1_concat_test_Acc, branch1_combine_test_Acc, branch2_concat_test_Acc,
                branch2_combine_test_Acc, last_combine_Acc, branch1_combine_loss / (batch_idx + 1), branch2_combine_loss / (batch_idx + 1),
                last_combine_loss / (batch_idx + 1)))

        test_loss=last_combine_loss/(idx + 1)
    return branch1_concat_test_Acc, branch1_combine_test_Acc, branch2_concat_test_Acc,branch2_combine_test_Acc,last_combine_Acc,test_loss
