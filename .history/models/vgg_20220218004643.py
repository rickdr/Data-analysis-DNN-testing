import torch
import torch.nn as nn
from torchvision import models

class VGGAutoencoder(torch.nn.Module):
    def __init__(self, model_name, pretrained):
        super(Model_Based_Autoencoder, self).__init__()
        if model_name != 'vgg':
            sys.stdout.write('Dear, we only support vgg now...')

        self.encoder = models.vgg11_bn(pretrained=pretrained).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv6
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv1
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_16(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if pre_trained and path is None:
        print("Pretrained model")
        model = models.vgg16(pretrained=pre_trained)
    else:
        if classes is None:
            model = models.vgg16()
        else:
            model = models.vgg16(num_classes=classes)

    if classes == 10:
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer, classes)

    if classes == 100:
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer, classes)

    if path is not None:
        print("Load pretrained model")
        model.load_state_dict(torch.load(path, map_location=device))

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"16", model


def load_19(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if pre_trained and path is not None:
        model = models.vgg19(num_classes=classes)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    else:
        model = models.vgg19(pretrained=pre_trained, num_classes=classes)

    if frozen:
        model = model.eval()

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    return model.__class__.__name__+"19", model