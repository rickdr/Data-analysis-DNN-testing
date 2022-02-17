import torch
from torchvision import models


def load_121(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if pre_trained and path is not None:
        model = models.densenet121(num_classes=classes)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    else:
        model = models.densenet121(pretrained=pre_trained, num_classes=classes)

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"121", model


def load_121(pre_trained=False, frozen=False, path=None, device=None, classes=10):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if pre_trained and path is not None:
        model = models.densenet161(num_classes=classes)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    else:
        model = models.densenet161(pretrained=pre_trained, num_classes=classes)

    if device == torch.device("cuda:0"):
        print("Cuda is enabled")
        model.cuda()

    if frozen:
        model = model.eval()

    return model.__class__.__name__+"161", model
