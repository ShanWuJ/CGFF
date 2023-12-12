from __future__ import print_function
import torch.optim as optim
from predict import *
from torch.utils.data import DataLoader
from utils import *


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    root='./Cub/train'
    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_msg', pretrain=True, require_grad=True)
    netp = torch.nn.DataParallel(net, device_ids=[0,1])
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device('cuda:0')
    netp.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.features.parameters(), 'lr': 0.0002},
        {'params': net.basic_block1.parameters(), 'lr': 0.002},
        {'params': net.basic_block2.parameters(), 'lr': 0.002},
        {'params': net.basic_block3.parameters(), 'lr': 0.002},
        {'params': net.branch1.parameters(), 'lr': 0.002},
        {'params': net.branch2.parameters(), 'lr': 0.002},
        {'params': net.classifier6.parameters(), 'lr': 0.002}
    ],
        momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    lr = [0.0002,0.002, 0.002, 0.002, 0.002,0.002, 0.002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        netp.train()
        branch1_combine_train_loss=0
        branch2_combine_train_loss=0
        last_combine_train_loss = 0
        branch1_correct_concat = 0
        branch2_correct_concat = 0
        branch1_correct_combine = 0
        branch2_correct_combine = 0
        high_level_correct = 0
        last_conbine_correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            # if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            optimizer.zero_grad()

            cls1,cls2,cls6 = netp(inputs)
            cls11, cls12, cls13, cls14, cls_concat1 = cls1
            cls21, cls22, cls23, cls24, cls_concat2 = cls2

            loss11 = CELoss(cls11, targets) * 1
            loss12 = CELoss(cls12, targets) * 1
            loss13 = CELoss(cls13, targets) * 1
            loss14 = CELoss(cls14, targets) * 1

            loss21 = CELoss(cls21, targets) * 1
            loss22 = CELoss(cls22, targets) * 1
            loss23 = CELoss(cls23, targets) * 1
            loss24 = CELoss(cls24, targets) * 1

            loss6 =  CELoss(cls6, targets) * 1

            branch1_concat_loss = CELoss(cls_concat1, targets) * 2
            branch2_concat_loss = CELoss(cls_concat2, targets) * 2
            branch1_combine_loss = loss11+loss12+loss13+loss14+branch1_concat_loss+loss6
            branch2_combine_loss = loss21+loss22+loss23+loss24+branch2_concat_loss+loss6
            loss=branch1_combine_loss+branch2_combine_loss

            loss.backward()
            optimizer.step()

            branch1_combine_output = cls11+cls12+cls13+cls14+cls_concat1+cls6
            branch2_combine_output = cls21+cls22+cls23+cls24+cls_concat2+cls6
            last_output = branch1_combine_output+branch2_combine_output+cls6

            _, branch1_predicted_concat = torch.max(cls_concat1.data, 1)
            _, branch2_predicted_concat = torch.max(cls_concat2.data, 1)
            _, branch1_predicted_combine = torch.max(branch1_combine_output.data, 1)
            _, branch2_predicted_combine = torch.max(branch2_combine_output.data, 1)
            _, last_predicted = torch.max(last_output,1)
            _, high_level_predict = torch.max(cls6,1)
            total += targets.size(0)

            branch1_correct_concat += branch1_predicted_concat.eq(targets.data).cpu().sum()
            branch2_correct_concat += branch2_predicted_concat.eq(targets.data).cpu().sum()
            branch1_correct_combine += branch1_predicted_combine.eq(targets.data).cpu().sum()
            branch2_correct_combine += branch2_predicted_combine.eq(targets.data).cpu().sum()
            high_level_correct += high_level_predict.eq(targets.data).cpu().sum()
            last_conbine_correct += last_predicted.eq(targets.data).cpu().sum()

            branch1_combine_train_loss += branch1_combine_loss.item()
            branch2_combine_train_loss += branch2_combine_loss.item()
            last_combine_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | branch1_comLoss: %.3f | branch2_comLoss: %.5f | '
                    'Last_Loss: %.5f | branch1_comAcc: %.5f |branch1_catAcc: %.5f | branch2_comAcc: %.3f | branch2_catAcc: %.3f | Last_Acc: %.3f | High_level_Acc: %.3f' % (
                    batch_idx, branch1_combine_train_loss / (batch_idx + 1),branch2_combine_train_loss / (batch_idx + 1),
                    last_combine_train_loss/(batch_idx+1),100. * float(branch1_correct_combine) / total,100. * float(branch1_correct_concat) / total,
                    100. * float(branch2_correct_combine) / total, 100. * float(branch2_correct_concat) / total,100. * float(last_conbine_correct) / total,100. * float(high_level_correct) / total))

        high_level_train_acc = 100. * float(high_level_correct) / total
        Branch1_combine_train_acc = 100. * float(branch1_correct_combine) / total
        Branch1_concat_train_acc = 100. * float(branch1_correct_concat) / total
        Branch2_combine_train_acc = 100. * float(branch2_correct_combine) / total
        Branch2_concat_train_acc = 100. * float(branch2_correct_concat) / total
        Last_combine_train_acc = 100. * float(last_conbine_correct) / total
        last_combine_train_loss = last_combine_train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Interation: %d | branch1_comLoss: %.3f | branch2_comLoss: %.5f | '
                'Last_Loss: %.5f | branch1_comAcc: %.5f |branch1_catAcc: %.5f | branch2_comAcc: %.3f | branch2_catAcc: %.3f | Last_Acc: %.3f| High_level_Acc: %.3f|\n' % (
                    epoch, branch1_combine_train_loss / (idx + 1),branch2_combine_train_loss / (idx + 1),
                    last_combine_train_loss ,Branch1_combine_train_acc, Branch1_concat_train_acc,
                    Branch2_combine_train_acc, Branch2_concat_train_acc, Last_combine_train_acc,high_level_train_acc))

        if epoch < 5 or epoch >= 60:
            # 在这里进行验证
            branch1_concat_test_Acc, branch1_combine_test_Acc, branch2_concat_test_Acc, branch2_combine_test_Acc, last_combine_Acc, test_loss = test(net, CELoss, 16)
            if last_combine_Acc > max_val_acc:
                max_val_acc = last_combine_Acc
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, branch1_concat_test_Acc: %.3f,branch1_combine_test_Acc: %.3f,'
                           'branch2_concat_test_Acc: %.3f,branch2_combine_test_Acc:%.3f,last_combine_Acc:%.3f,last_loss:%.6f\n' % (
                               epoch, branch1_concat_test_Acc, branch1_combine_test_Acc, branch2_concat_test_Acc,
                               branch2_combine_test_Acc, last_combine_Acc, test_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)

# 训练方法的调用
train(nb_epoch=200,  # number of epoch
      batch_size=32,  # batch size
      store_name='bird',  # folder for output
      resume=False,  # resume training from checkpoint
      start_epoch=0,  # the start epoch number when you resume the training
      model_path='')  # the saved model where you want to resume the training
