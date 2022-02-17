import torch
import torch.nn as nn

class LinearSVM(nn.Module):
    def __init__(self,n_feature,n_class):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(n_feature,n_class)
        torch.nn.init.kaiming_uniform(self.fc.weight)
        torch.nn.init.constant(self.fc.bias,0.1)
        
    def forward(self,x):
        output=self.fc(x)
        return output


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True, lr=0):
        super(multiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight #weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average = size_average

    def forward(self, output, y): #output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y = output[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()].view(-1, 1)#view for transpose
        #margin - output[y] + output[i]
        loss = output - output_y + self.margin #contains i=y
        #remove i=y items
        loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()] = 0
        #max(0,_)
        loss[loss < 0] = 0
        #^p
        if(self.p != 1):
            loss = torch.pow(loss, self.p)
        #add weight
        if(self.weight is not None):
            loss = loss * self.weight
        #sum up
        loss = torch.sum(loss)
        if(self.size_average):
            loss /= output.size()[0]#output.size()[0]
        return loss


def load_Linear(pre_trained=False, frozen=False, path=None, device=None):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = LinearSVM(784, 10)

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
        # model.to(torch.device('cuda'))

    return model.__class__.__name__, model
