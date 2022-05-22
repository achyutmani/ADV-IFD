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

Fault_Model=resnet18()
Fault_Model=Fault_Model.to(device)
Fault_Model_optimizer = optim.Adam(Fault_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc 
def train(model,device,iterator, optimizer, criterion): # Define Training Function 
    #early_stopping = EarlyStopping(patience=7, verbose=True)
    #print("Training Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.train() # call model object for training 
    
    for (x,y) in iterator:
        x=x.float()
        x=x.to(device)
        y=y.to(device)# Transfer label  to device
        y=y.long()
        optimizer.zero_grad() # Initialize gredients as zeros 
        count=count+1
        #print(x.shape)
        Predicted_Train_Label=model(x)
        #print(Predicted_Train_Label)
        #print(y)
        loss = criterion(Predicted_Train_Label, y) # training loss
        acc = calculate_accuracy(Predicted_Train_Label, y) # training accuracy 
        #print("Training Iteration Number=",count)
        loss.backward() # backpropogation 
        optimizer.step() # optimize the model weights using an optimizer 
        epoch_loss += loss.item() # sum of training loss
        epoch_acc += acc.item() # sum of training accuracy  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    
    with torch.no_grad(): # Without computation of gredient 
        for (x, y) in iterator:
            x=x.float()
            x=x.to(device) # Transfer data to device 
            y=y.to(device) # Transfer label  to device 
            y=y.long()
            count=count+1
            Predicted_Label = model(x) # Predict claa label 
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
            #print("Validation Iteration Number=",count)
            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy 
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_ResNet.pt') # Define Path to save the model 
best_valid_loss = float('inf')
Temp=np.zeros([EPOCHS,6]) # Temp Matrix to Store all model accuracy, loss and time parameters 
print("Fault CNN Model is in Training Mode") 
print("---------------------------------------------------------------------------------------------------------------------")   
#early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc = train(Fault_Model,device,train_loader,Fault_Model_optimizer, criterion) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc = evaluate(Fault_Model,device,train_loader,criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    print(" | Epoch=",epoch," | Training Accuracy=",train_acc*100," | Validation Accuracy=",valid_acc*100," | Training Loss=",train_loss," | Validation_Loss=",valid_loss,"Time Taken(Seconds)=",end_time,"|")
    print("---------------------------------------------------------------------------------------------------------------------")
    Temp[epoch,0]=epoch # Store Epoch Number 
    Temp[epoch,1]=train_acc # Store Training Accuracy 
    Temp[epoch,2]=valid_acc # Store Validation Accuracy 
    Temp[epoch,3]=train_loss # Store Training Loss 
    Temp[epoch,4]=valid_loss # Store Validation Loss 
    Temp[epoch,5]=end_time # Store Running Time of One Epoch 
    #early_stopping(valid_loss,Teacher_Model) # call Early Stopping to Prevent Overfitting 
    #if early_stopping.early_stop:
        #print("Early stopping")
        #break
    #Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
torch.save(Fault_Model.state_dict(), MODEL_SAVE_PATH)    
np.save('Fault_CNN1_Parameters',Temp) # Save Temp Array as numpy array 
Fault_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc = evaluate(Fault_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
#test_loss=round(test_loss,2)# Round test loss
#test_acc=round(test_acc,2) # Round test accuracy 
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100) # print test accuracy 
#print(Test_CM)
import itertools
import numpy as np
import matplotlib.pyplot as plt
def get_all_preds(model,loader):
     all_preds = torch.tensor([])
     all_preds=all_preds.to(device)
     all_actual=torch.tensor([])
     all_actual=all_actual.to(device)
     for batch in loader:
         images, labels = batch
         images=images.to(device)
         images=images.float()
         labels=labels.to(device)
         labels=labels.long()  
         preds = (nn.functional.softmax(model(images),dim=1)).max(1,keepdim=True)[1]
         #fx.max(1, keepdim=True)[1]
#         #print(preds)
#         #print(labels)
#         #dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
         all_preds = torch.cat((all_preds, preds.float()),dim=0)
#         #print(all_preds)
         all_actual = torch.cat((all_actual,labels),dim=0)
     return all_preds,all_actual
test_preds,test_actual = get_all_preds(Fault_Model,test_loader) 
Test_CM = confusion_matrix(test_actual.cpu().numpy(),test_preds.cpu().numpy())    
classes= ['Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'Normal',
 'OR_007', 'OR_014', 'OR_021']
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,size=12)
    plt.yticks(tick_marks, classes,rotation=45,size=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',size=12)
    plt.xlabel('Predicted label',size=12)
plt.figure(figsize=(7,7))
plt.rcParams['font.size'] = 12
plot_confusion_matrix(Test_CM,classes) 
plt.tight_layout()
plt.savefig('Fault_ResNet_CM.png')
plt.show()
