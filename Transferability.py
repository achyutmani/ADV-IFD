import torch 
import os 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torch.optim import Adam
import torchaudio
import pandas as pd
import numpy as np
import time # import time 
import sys # Import System 
from torch import optim, cuda # import optimizer  and CUDA
import random
import torch.nn.functional as F
from FD_Custom_Dataloader import TSData_Train
from sklearn.metrics import confusion_matrix
Num_Class=10
learning_rate=0.0008
batch_size=128
SEED = 1234 # Initialize seed 
EPOCHS=100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor()])
train_dataset=TSData_Train(transform=train_transformations)
train_size = int(0.8 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Test_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
#valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
class LeNet(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(LeNet, self).__init__()
        self.CNN1=nn.Conv2d(1,6,kernel_size=(5,5))
        self.MP1=nn.MaxPool2d(kernel_size=2)
        self.CNN2=nn.Conv2d(6,16,kernel_size=(5,5))
        self.MP2=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,Num_Class)
    def forward(self,TS):
        x1=self.CNN1(TS)
        x1=F.relu(x1)
        x2=self.MP1(x1)
        x3=self.CNN2(x2)
        x3=F.relu(x3)
        x4=self.MP2(x3)
        #print(x4.shape)
        #x5=x4.view(x4.size(0),-1)
        x5=torch.flatten(x4, start_dim=1)
        #print(x5.shape)
        x6=self.fc1(x5)
        x7=self.fc2(x6)
        x8=self.fc3(x7)
        #x8=F.softmax(x8)
        #print(x8)
        return x8    
class CNN_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.CNN1=nn.Conv2d(1,32,kernel_size=(9,9))
        self.MP1=nn.MaxPool2d(kernel_size=2)
        self.CNN2=nn.Conv2d(32,32,kernel_size=(9,9))
        self.MP2=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,96)
        self.fc3=nn.Linear(96,Num_Class)
    def forward(self,TS):
        x1=self.CNN1(TS)
        x1=F.relu(x1)
        x2=self.MP1(x1)
        x3=self.CNN2(x2)
        x3=F.relu(x3)
        x4=self.MP2(x3)
        #print(x4.shape)
        #x5=x4.view(x4.size(0),-1)
        x5=torch.flatten(x4, start_dim=1)
        #print(x5.shape)
        x6=self.fc1(x5)
        x7=self.fc2(x6)
        x8=self.fc3(x7)
        #x8=F.softmax(x8)
        #print(x8)
        return x8 
class AlexNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_channel),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x              
#Sounce_Model=CNN_Model()
#Sounce_Model=AlexNet()
#Sounce_Model=LeNet()
Sounce_Model=resnet18()
#Target_Model=CNN_Model()
#Target_Model=AlexNet()
Target_Model=LeNet()
#Target_Model=resnet18()
Source_Model=Sounce_Model.to(device)
Target_Model=Target_Model.to(device)
#Fault_Model_optimizer = optim.Adam(Fault_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc  
#Source_Model1_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN1.pt') # Define Path to save the model 
#Source_Model2_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_AlexNet_1D.pt') # Define Path to save the model 
#Source_Model3_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN3s.pt') # Define Path to save the model 
Source_Model4_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_ResNet.pt')
#Target_Model1_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN1.pt') # Define Path to save the model 
#Target_Model2_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_AlexNet_1D.pt') # Define Path to save the model 
Target_Model3_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN3s.pt') # Define Path to save the model 
#Target_Model4_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_ResNet.pt')
#Source_Model.load_state_dict(torch.load(Source_Model1_SAVE_PATH)) 
#Source_Model.load_state_dict(torch.load(Source_Model2_SAVE_PATH))
#Source_Model.load_state_dict(torch.load(Source_Model3_SAVE_PATH))
Source_Model.load_state_dict(torch.load(Source_Model4_SAVE_PATH))
#Target_Model.load_state_dict(torch.load(Target_Model1_SAVE_PATH))
#Target_Model.load_state_dict(torch.load(Target_Model2_SAVE_PATH))
Target_Model.load_state_dict(torch.load(Target_Model3_SAVE_PATH))
#Target_Model.load_state_dict(torch.load(Target_Model4_SAVE_PATH))

import torchattacks
#attack=torchattacks.BIM(Source_Model, 32/255,1/255,0)
#attack=torchattacks.FGSM(Source_Model,eps=32/255)
#attack=torchattacks.PGD(Source_Model,32/255,1/255)
attack=torchattacks.CW(Source_Model,c=1, kappa=0, lr=0.01)
def evaluate1(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    ADV_Dist=0
    all_preds = torch.tensor([])
    all_preds=all_preds.to(device)
    model.eval() # call model object for evaluation 
    #with torch.no_grad(): # Without computation of gredient 
    for (x,y) in iterator:
        x=x.float()
        x=x.to(device) # Transfer data to device 
        y=y.to(device) # Transfer label  to device 
        y=y.long()
        count=count+1
        adv_images=attack(x,y)
        adv_images=adv_images.to(device)
        L2_Dist=torch.norm(torch.abs(adv_images-x))
        x=x.detach()
        Predicted_Label = model(adv_images) # Predict claa label
        preds = (nn.functional.softmax(model(adv_images),dim=1)).max(1,keepdim=True)[1]
        all_preds = torch.cat((all_preds, preds.float()),dim=0) 
        loss = criterion(Predicted_Label, y) # Compute Loss 
        acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
        #print("Validation Iteration Number=",count)
        epoch_loss += loss.item() # Compute Sum of  Loss 
        epoch_acc += acc.item() # Compute  Sum of Accuracy
        ADV_Dist=ADV_Dist+L2_Dist   
    return epoch_loss / len(iterator), epoch_acc / len(iterator),all_preds, ADV_Dist/len(iterator), all_preds 
def Class_Distribution(Class_Dist):
    arr=Class_Dist.detach().cpu().numpy()
    uniqueValues, occurCount = np.unique(arr, return_counts=True)
    occurCount=(occurCount/len(arr))*100
    print("Unique Classes=",uniqueValues)
    print("Class Distribution=",occurCount)
test_loss, test_acc,Class_Dist, ADV_Dist,Adv_test_preds = evaluate1(Target_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
#Class_Distribution(Class_Dist)
test_acc=100-(test_acc*100)
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc)

