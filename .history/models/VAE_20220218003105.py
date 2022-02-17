
# # class VAE(nn.Module):
# #     def __init__(self):
# #         super(VAE, self).__init__()

# #         # Encoder
# #         self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
# #                                stride=1, padding=1, bias=False)
# #         self.bn1 = nn.BatchNorm2d(16)
# #         self.conv2 = nn.Conv2d(16, 32, kernel_size=3,
# #                                stride=2, padding=1, bias=False)
# #         self.bn2 = nn.BatchNorm2d(32)
# #         self.conv3 = nn.Conv2d(32, 32, kernel_size=3,
# #                                stride=1, padding=1, bias=False)
# #         self.bn3 = nn.BatchNorm2d(32)
# #         self.conv4 = nn.Conv2d(32, 16, kernel_size=3,
# #                                stride=2, padding=1, bias=False)
# #         self.bn4 = nn.BatchNorm2d(16)

# #         self.fc1 = nn.Linear(8 * 8 * 16, 512)
# #         self.fc_bn1 = nn.BatchNorm1d(512)
# #         self.fc21 = nn.Linear(512, 512)
# #         self.fc22 = nn.Linear(512, 512)

# #         # Decoder
# #         self.fc3 = nn.Linear(512, 512)
# #         self.fc_bn3 = nn.BatchNorm1d(512)
# #         self.fc4 = nn.Linear(512, 8 * 8 * 16)
# #         self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

# #         self.conv5 = nn.ConvTranspose2d(
# #             16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
# #         self.bn5 = nn.BatchNorm2d(32)
# #         self.conv6 = nn.ConvTranspose2d(
# #             32, 32, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn6 = nn.BatchNorm2d(32)
# #         self.conv7 = nn.ConvTranspose2d(
# #             32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
# #         self.bn7 = nn.BatchNorm2d(16)
# #         self.conv8 = nn.ConvTranspose2d(
# #             16, 3, kernel_size=3, stride=1, padding=1, bias=False)

# #         self.relu = nn.SELU()

# #     def encode(self, x):
# #         conv1 = self.relu(self.bn1(self.conv1(x)))
# #         conv2 = self.relu(self.bn2(self.conv2(conv1)))
# #         conv3 = self.relu(self.bn3(self.conv3(conv2)))
# #         conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

# #         fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
# #         return self.fc21(fc1), self.fc22(fc1)

# #     def reparameterize(self, mu, logvar):
# #         if self.training:
# #             std = logvar.mul(0.5).exp_()
# #             eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
# #             return eps.mul(std).add_(mu)
# #         else:
# #             return mu

# #     def decode(self, z):
# #         fc3 = self.relu(self.fc_bn3(self.fc3(z)))
# #         fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

# #         conv5 = self.relu(self.bn5(self.conv5(fc4)))
# #         conv6 = self.relu(self.bn6(self.conv6(conv5)))
# #         conv7 = self.relu(self.bn7(self.conv7(conv6)))
# #         # return self.conv8(conv7).view(-1, 3, 32, 32)
# #         return self.conv8(conv7)
        
# #     def forward(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return (mu, logvar), self.decode(z).view(-1, 3, 32, 32)

# #     def latent(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return z

# #     def sample(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return (mu, logvar), torch.sigmoid(self.decode(z))

#     # def sample(self, n_samples):
#     #     latent_size = self._h // 2 ** (self._total_stride // 2)
#     #     shape = (n_samples, self._latent_channels, latent_size, latent_size)
#     #     latents = torch.randn(shape, device=self.device)
#     #     return self._decoder(latents)

# # class new_VAE(nn.Module):
# #     def __init__(self, hidden_dim=64):
# #         super(new_VAE, self).__init__()

# #         self.hidden_dim = hidden_dim
# #         self.encoder = nn.Sequential(
# #             nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
# #             nn.BatchNorm2d(16),
# #             nn.ReLU(),
# #             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(),
# #         )

# #         self.q_mean = nn.Linear(4 * 4 * 64, hidden_dim)
# #         self.q_logvar = nn.Linear(4 * 4 * 64, hidden_dim)
# #         self.project = nn.Linear(hidden_dim, 4 * 4 * 64)

# #         self.decoder = nn.Sequential(
# #             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
# #             nn.BatchNorm2d(16),
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
# #         )

# #     def forward(self, x):
# #         x = self.encoder(x)

# #         x = x.view(x.shape[0], -1)
# #         mu = self.q_mean(x)
# #         logvar = self.q_logvar(x)
# #         z = self.reparameterize(mu, logvar)
# #         z = self.project(z).view(z.shape[0], -1, 4, 4)
# #         z = self.decoder(z)
# #         return (mu, logvar), z

# #     def encode(self, x):
# #         x = self.encoder(x)
# #         x = x.view(x.shape[0], -1)
# #         mu = self.q_mean(x)
# #         logvar = self.q_logvar(x)
# #         z = self.reparameterize(mu, logvar)
# #         z = self.project(z).view(z.shape[0], -1, 4, 4)
# #         return z

# #     def reparameterize(self, mu, logvar):
# #         std = torch.exp(0.5 * logvar)
# #         eps = torch.randn_like(std)
# #         return eps * std + mu

# #     def latent(self, x):
# #         (mu, logvar), z = self.forward(x)
# #         return z

# #     def sample(self, x):
# #         (mu, logvar), z = self.forward(x)
# #         return torch.sigmoid(z)

# # class VAE_Cifar(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         # Encoder
# #         self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
# #         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
# #         self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
# #         self.fc1 = nn.Linear(16 * 16 * 32, 128)

# #         # Latent space
# #         self.fc21 = nn.Linear(128, 20)
# #         self.fc22 = nn.Linear(128, 20)

# #         # Decoder
# #         self.fc3 = nn.Linear(20, 128)
# #         self.fc4 = nn.Linear(128, 8192)
# #         self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
# #         self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
# #         self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
# #         self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

# #         self.relu = nn.SELU()

# #     def encode(self, x):
# #         out = self.relu(self.conv1(x))
# #         out = self.relu(self.conv2(out))
# #         out = self.relu(self.conv3(out))
# #         out = self.relu(self.conv4(out))
# #         out = out.view(out.size(0), -1)
# #         h1 = self.relu(self.fc1(out))
# #         return self.fc21(h1), self.fc22(h1)

# #     def reparameterize(self, mu, logvar):
# #         """ logvar is the log of the variance. """
# #         if self.training:
# #             # std = exp(1/2 * logvar)
# #             std = logvar.mul(0.5).exp_()
# #             # eps ~ N(0, std)
# #             eps = std.data.new(std.size()).normal_()
# #             # retval ~ N(mu, std)
# #             return eps.mul(std).add_(mu)
# #         else:
# #             return mu

# #     def decode(self, z):
# #         h3 = self.relu(self.fc3(z))
# #         out = self.relu(self.fc4(h3))
# #         # import pdb; pdb.set_trace()
# #         out = out.view(out.size(0), 32, 16, 16)
# #         out = self.relu(self.deconv1(out))
# #         out = self.relu(self.deconv2(out))
# #         out = self.relu(self.deconv3(out))
# #         out = self.conv5(out)
# #         return out

# #     def forward(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return (mu, logvar), self.decode(z)

# #     def latent(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return z

# #     def sample(self, x):
# #         mu, logvar = self.encode(x)
# #         z = self.reparameterize(mu, logvar)
# #         return (mu, logvar), torch.sigmoid(self.decode(z))




# def load_vae(pre_trained=False, frozen=False, path=None, device=None, classes=10):
#     if device is None:
#         device = torch.device(
#             "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#     model = VAE()

#     if pre_trained:
#         if path is not None:
#             model.load_state_dict(torch.load(
#                 path, map_location=torch.device(device)))
#         else:
#             print("Specify a path to the model that needs to be loaded.")
#             return "", None

#     if device == torch.device("cuda:0"):
#         print("Cuda is enabled")
#         model.cuda()

#     if frozen:
#         model = model.eval()

#     return model.__class__.__name__, model


# def load_cifar_vae(pre_trained=False, frozen=False, path=None, device=None, classes=10):
#     if device is None:
#         device = torch.device(
#             "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#     model = VAE_Cifar()

#     if pre_trained:
#         if path is not None:
#             model.load_state_dict(torch.load(
#                 path, map_location=torch.device(device)))
#         else:
#             print("Specify a path to the model that needs to be loaded.")
#             return "", None

#     if device == torch.device("cuda:0"):
#         print("Cuda is enabled")
#         model.cuda()

#     if frozen:
#         model = model.eval()

#     return model.__class__.__name__, model


# def load_new_vae(pre_trained=False, frozen=False, path=None, device=None, classes=10):
#     if device is None:
#         device = torch.device(
#             "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#     model = new_VAE()

#     if pre_trained:
#         if path is not None:
#             model.load_state_dict(torch.load(
#                 path, map_location=torch.device(device)))
#         else:
#             print("Specify a path to the model that needs to be loaded.")
#             return "", None

#     if device == torch.device("cuda:0"):
#         print("Cuda is enabled")
#         model.cuda()

#     if frozen:
#         model = model.eval()

#     return model.__class__.__name__, model
