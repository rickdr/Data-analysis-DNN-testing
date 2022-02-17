import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://github.com/me2140733/AKI/blob/1e2e84b86b6a87726a48fd43594709e667046953/pl_bolts/models/autoencoders/basic_ae/basic_ae_module.py

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

# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()

#         # Encoder
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         # Decoder
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         return x
    
#     def decode(self, z):
#         z = self.pool(z)
#         z = F.relu(self.t_conv1(z))
#         z = torch.sigmoid(self.t_conv2(z))
#         return z

#     def latent(self, x):
#         z = self.pool(x)
#         z = F.relu(self.t_conv1(z))
#         z = self.t_conv2(z)
#         return z

#     def forward(self, x):
#         z = self.encode(x)
#         x = self.decode(z)
#         return x

 #https://github.com/jorgemarpa/PPDAE/blob/954522340d6b337b6f2cedb667c57f85ecc0f390/src/.ipynb_checkpoints/ae_model-checkpoint.py
# class SimpleAutoencoder(nn.Module):
#     def __init__(self):
#         super(SimpleAutoencoder, self).__init__()
#         self.latent_dim = 32
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

#             nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

#             Flatten(),
#             nn.Linear(512, self.latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_dim, 512),
#             nn.ReLU(),
#             Reshape(32, 4, 4),

#             nn.MaxUnpool2d(kernel_size=2, stride=2),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             nn.MaxUnpool2d(kernel_size=2, stride=2),
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),

#             nn.MaxUnpool2d(kernel_size=2, stride=2),
#             nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         return x
    
#     def decode(self, z):
#         z = self.decoder(z)
#         return z

#     def latent(self, x):
#         return self.encode(x)

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# class AE(nn.Module):
#     def __init__(self):
#         super(AE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 6, kernel_size=5),     # 28 * 28 * 6
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, kernel_size=5),    # 24 * 24 * 16
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, kernel_size=5),   # 20 * 20 * 32 
#             nn.ReLU(True)
#             # nn.Dropout2d(0.5),                  # 10 * 10 * 32
#             )
#         self.latent = nn.Linear(32 * 20 * 20, 10)
#         self.latent2 = nn.Linear(10, 32 * 20 * 20)

#         self.decoder = nn.Sequential(
#             # nn.Upsample(2),
#             nn.ConvTranspose2d(32, 16, kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(6, 3, kernel_size=5),
#             nn.ReLU(True),
#             )

#         self.getLatent = False

#     def forward(self, x):
#         batch = x.shape[0]
#         x = self.encoder(x)
#         x = x.view(batch, -1)
#         x = self.latent(x)

#         if self.getLatent:
#             return x
        
#         x = self.latent2(x)
#         x = x.view(batch, 32 * 20 * 20)
#         x = x.view(-1, 32, 20, 20)
#         x = self.decoder(x)
#         return x

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

# class ConvAE(nn.Module):
#     def __init__(self):
#         super(ConvAE, self).__init__()

#         self.latent_size = 128

#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             Flatten(),
#             nn.Linear(6272, 1024),
#             nn.ReLU()
#         )

#         # hidden => z
#         self.fc1 = nn.Linear(1024, self.latent_size)

#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_size, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 6272),
#             nn.ReLU(),
#             Unflatten(128, 7, 7),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def encode(self, x):
#         z = self.encoder(x)
#         return z

#     def decode(self, z):
#         z = self.decoder(z)
#         return z

#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z)

# define the NN architecture
# class LangeConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(LangeConvAutoencoder, self).__init__()
#         self.flatten = Flatten()  # describing the layer

#         representation_size = 14

#         # encoder layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=1,stride=2)

#         self.enc_f1 = nn.Linear(in_features=2704, out_features=144)
#         self.enc_f2 = nn.Linear(in_features=144, out_features=72)
#         self.enc_f3 = nn.Linear(in_features=72, out_features=36)
#         self.enc_f4 = nn.Linear(in_features=36, out_features=18)
#         self.enc_f5 = nn.Linear(in_features=18, out_features=representation_size)

#         # decoder layers
#         self.dec_f1 = nn.Linear(in_features=representation_size, out_features=18)
#         self.dec_f2 = nn.Linear(in_features=18, out_features=36)
#         self.dec_f3 = nn.Linear(in_features=36, out_features=72)
#         self.dec_f4 = nn.Linear(in_features=72, out_features=144)
#         self.dec_f5 = nn.Linear(in_features=144, out_features=2704)

#         self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=7, padding=1,stride=2,output_padding=1)
#         self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=7,padding=1)
#         self.t_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=7, padding=1)

#     def forward(self, x):
#         ## encode ##
#         # add hidden layers with relu activation function
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         unflatten_size = x.shape
#         x = self.flatten(x)

#         x = F.relu(self.enc_f1(x))
#         x = F.relu(self.enc_f2(x))
#         x = F.relu(self.enc_f3(x))
#         x = F.relu(self.enc_f4(x))
#         x = F.relu(self.enc_f5(x))

#         representation = x

#         ## decode ##
#         x = F.relu(self.dec_f1(x))
#         x = F.relu(self.dec_f2(x))
#         x = F.relu(self.dec_f3(x))
#         x = F.relu(self.dec_f4(x))
#         x = F.relu(self.dec_f5(x))
#         x = x.reshape(unflatten_size)

#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))

#         return representation, x

class ConvAutoencoderCIFAR(nn.Module):
    def __init__(self, latent_size=100):
        super(ConvAutoencoderCIFAR, self).__init__()
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

import logging
import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

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

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             nn.ReLU(),
# 			# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             # nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             # nn.ReLU(),
# 			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         return x
    
#     def decode(self, z):
#         z = self.decoder(z)
#         return z

#     def latent(self, x):
#         return self.encode(x)

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

# class AE_Cifar(nn.Module):
#     def __init__(self):
#         super(AE_Cifar, self).__init__()

#         # Encoder
#         self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(16 * 16 * 32, 128)

#         # Latent space
#         self.fc21 = nn.Linear(128, 32)

#         # Decoder
#         self.fc3 = nn.Linear(32, 128)
#         self.fc4 = nn.Linear(128, 8192)
#         self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
#         self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

#         self.relu = nn.SELU()

#     def encode(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.relu(self.conv3(out))
#         out = self.relu(self.conv4(out))
#         out = out.view(out.size(0), -1)
#         h1 = self.relu(self.fc1(out))
#         out = self.fc21(h1)
#         return out

#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         out = self.relu(self.fc4(h3))
#         # import pdb; pdb.set_trace()
#         out = out.view(out.size(0), 32, 16, 16)
#         out = self.relu(self.deconv1(out))
#         out = self.relu(self.deconv2(out))
#         out = self.relu(self.deconv3(out))
#         out = self.conv5(out)
#         return out

#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z)

#     def latent(self, x):
#         return self.encode(x)

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

def load_ConvAutoencoderCIFAR(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = ConvAutoencoderCIFAR()

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