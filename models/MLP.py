import torch
import torch.nn as nn
import torch.nn.functional


class MLP_three_layered(nn.Module):
    def __init__(self):
        super(three_layered, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.droput = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = self.droput(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x


def load_3_layered(pre_trained=False, frozen=False, path=None, device=None):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = MLP_three_layered()

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
