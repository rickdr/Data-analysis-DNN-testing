import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class Unflatten(nn.Module):

    def __init__(self, C, H, W):
        super(Unflatten, self).__init__()
        self.N = -1
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class Flatten(nn.Module):
    """
    Examples
    --------
    >>> m = Flatten()
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = m(x)
    >>> y.size()
    torch.Size([32, 150])
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class View(nn.Module):
    """
    Examples
    --------
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = View(-1)(x)
    >>> y.size()
    torch.Size([4800])
    >>> View(32, -1)(x).size()
    torch.Size([32, 150])
    >>> View(48, 10, 10)(y).size()
    torch.Size([48, 10, 10])
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Reshape(nn.Module):
    """
    Examples
    --------
    >>> x = torch.randn(32, 10, 5, 3)
    >>> y = Reshape(-1)(x)
    >>> y.size()
    torch.Size([32, 150])
    >>> Reshape(6, 5, 5)(y).size()
    torch.Size([32, 6, 5, 5])
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)


class MNIST_AutoEncoder(nn.Module):
    def __init__(self, code_size=20):
        super().__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out

class CIFAR_AutoEncoder(nn.Module):
    def __init__(self, latent_size=100):
        super(CIFAR_AutoEncoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
        # conv layer (depth from 16 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # conv layer (depth from 32 --> 4), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # dense layers
        self.fc1 = nn.Linear(256, 512) #flattening (input should be calculated by a forward pass - stupidity of Pytorch)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 32)

        ## decoder layers ##
        # decoding dense layer
        # self.dec_linear_1 = nn.Linear(32, 64)
        # self.dec_linear_2 = nn.Linear(64, 128)
        # self.dec_linear_3 = nn.Linear(128, 256)
        self.dec_linear_4 = nn.Linear(256, 512)
        self.dec_linear_5 = nn.Linear(512, 256)
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 32, 1, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x, return_comp=True):
        ## ==== encode ==== ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # add second hidden layer
        x = F.relu(self.conv3(x))
        # x = self.pool(x)  
        # flatten and apply dense layer
        x = x.view(-1, 8*8*4)
        x_comp = F.relu(self.fc1(x))
        x_comp = F.relu(self.fc2(x_comp))
        # x_comp = F.relu(self.fc3(x_comp))
        # x_comp = F.relu(self.fc4(x_comp))
        # x_comp = F.relu(self.fc5(x_comp))
        
        ## ==== decode ==== ##
        # x_comp = self.dec_linear_1(x_comp)
        # x_comp = self.dec_linear_2(x_comp)
        # x_comp = self.dec_linear_3(x_comp)
        x_comp = self.dec_linear_4(x_comp)
        x = self.dec_linear_5(x_comp)
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x.view(-1, 4, 8, 8)))
        x = F.relu(self.t_conv2(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv3(x))
        
        if return_comp:
            return x_comp, x 
        else:
            return x

    def latent(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # add second hidden layer
        x = F.relu(self.conv3(x))
        # x = self.pool(x)  
        # flatten and apply dense layer
        x = x.view(-1, 8*8*4)
        x_comp = F.relu(self.fc1(x))
        x_comp = F.relu(self.fc2(x_comp))
        return x_comp
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        intermediate_size = 256
        hidden_size = 256
        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def latent(self, x):
        return self.encode(x)

class CAE(nn.Module):
    def __init__(self, in_channels=3, rep_dim=32):
        super(CAE, self).__init__()
        slope = 0
        nf = 32
        self.nf = nf

        # Encoder: 32x32-x_channel-16x16x32-8x8x64-4x4x128-32
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.LeakyReLU(slope, inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.LeakyReLU(slope, inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.LeakyReLU(slope, inplace=True)

        self.enc_fc = nn.Linear(nf * 2 * 2 * 16, rep_dim)

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 2 * 2 * 16)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 2 * 2 * 16)
        self.dec_act0 = nn.LeakyReLU(slope, inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.LeakyReLU(slope, inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.LeakyReLU(slope, inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.enc_fc(x.view(x.size(0), -1))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        output = self.decode(self.encode(x))
        return output


import logging
import torch.nn as nn
import numpy as np

class CIFAR10_LeNet_Autoencoder(BaseNet):
    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x

def load_cifar_LenetAE(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = CIFAR10_LeNet_Autoencoder()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(
                path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__, model

def load_CIFAR_AutoEncoder(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = CIFAR_AutoEncoder()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(
                path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__, model

def load_cae_new(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = CAE()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(
                path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__, model

def load_cae(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # model = AE()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(
                path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__, model


def load_ae(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # model = basic_AE()
    model = AE()

    if pre_trained:
        if path is not None:
            model.load_state_dict(torch.load(
                path, map_location=torch.device(device)))
        else:
            print("Specify a path to the model that needs to be loaded.")
            return "", None

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__, model