import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_two_layered(nn.Module):
    def __init__(self):
        super(CNN_two_layered, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),      
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


def load_2_layered(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = CNN_two_layered()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if frozen:
        model = model.eval()

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()
        model.to(torch.device('cuda'))

    return model.__class__.__name__, model


class basic_CNN(nn.Module):
    def __init__(self, num_classes):
        super(basic_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, 1, 0)
        self.dense = torch.nn.Linear(512, 128)
        self.maxpool = torch.nn.MaxPool2d(2, 2, 0)
        self.outlayer = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = out.view((out.size(0), -1))
        out = self.dense(out)
        out = F.relu(out)
        out = self.outlayer(out)

        return out


def load_basic(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = basic_CNN(classes)

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if frozen:
        model = model.eval()

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()
        model.to(torch.device('cuda'))

    return model.__class__.__name__, model


class three_layered_CNN(nn.Module):
    def __init__(self):
        super(three_layered_CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)

        return x


def load_three_layered(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = three_layered_CNN()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if frozen:
        model = model.eval()

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()
        model.to(torch.device('cuda'))

    return model.__class__.__name__, model


class four_layered_CNN(nn.Module):
    def __init__(self):
        super(four_layered_CNN, self).__init__()
        self.module_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.module_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.Conv2d(64,64, 3, 1, 1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.module_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Dropout2d(p = 0.1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.module_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Dropout2d(p = 0.1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        x = self.module_4(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(2)
        
        x = x.view(x.size(0),-1)
        output = self.out(x)
        output = torch.nn.functional.log_softmax(output)
        return output


def load_four_layered(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = four_layered_CNN()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if frozen:
        model = model.eval()

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()
        model.to(torch.device('cuda'))

    return model.__class__.__name__, model
