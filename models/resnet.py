import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from torchvision import models


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])


def load_18(pre_trained=False, frozen=False, path=None, device=None, classes=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pre_trained and path is None:
        print("Pretrained model")
        model = models.resnet18(pretrained=pre_trained)
    else:
        if classes is None:
            model = models.resnet18()
        else:
            model = models.resnet18(num_classes=classes)

    if classes == 10:
        print("10 classes")
        model.fc.out_features = classes
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    if classes == 100:
        print("100 classes")
        model.fc.out_features = classes
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    # 200
    # model[1].avgpool = nn.AdaptiveAvgPool2d(1)
    # model[1].fc = torch.nn.Linear(model[1].fc.in_features, 200)

    if path is not None:
        print("Load pretrained model")
        model.load_state_dict(torch.load(path, map_location=device))

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"18", model


def load_20(pre_trained=False, frozen=False, path=None, device=None, classes=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet20()
    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"20", model


def load_50(pre_trained=False, frozen=False, path=None, device=None, classes=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pre_trained and path is None:
        model = models.resnet50(pretrained=pre_trained)
    else:
        if classes is None:
            model = models.resnet50()
        else:
            model = models.resnet50(num_classes=classes)

    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"50", model


def load_56(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet56()
    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"56", model
