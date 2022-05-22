import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
file = np.load('/home/mani/Desktop/IJCNN 2022/Bear Fault Detection/cbm_codes_open-master/notebooks/data/CWRU_48k_load_1_CNN_data.npz') # Give path to downloaded file in your system
#print(file.files)
data = file['data']
labels = file['labels']
print(np.unique(labels))
print(len(labels))
#print(data.shape, labels.shape)
#print(data[0])
#print(labels)
for i in range(0,len(labels)):
	if labels[i]=='Ball_007':
		labels[i]=0
	if labels[i]=='Ball_014':
		labels[i]=1
	if labels[i]=='Ball_021':
		labels[i]=2
	if labels[i]=='IR_007':
		labels[i]=3
	if labels[i]=='IR_014':
		labels[i]=4
	if labels[i]=='IR_021':
		labels[i]=5
	if labels[i]=='Normal':
		labels[i]=6
	if labels[i]=='OR_007':
		labels[i]=7
	if labels[i]=='OR_014':
		labels[i]=8
	if labels[i]=='OR_021':
		labels[i]=9									
#print(category_labels)
category_labels = np.unique(labels)
print(category_labels)
labels = pd.Categorical(labels, categories = category_labels).codes
class TSData_Train():
	def __init__(self,transform=None):
		self.annotations=data
		self.Label=labels
		self.Label=np.array(self.Label)
		self.transform=transform
	def __len__(self):
		return len(self.annotations)
	def __getitem__(self,index):
		TS_Data=np.array(self.annotations[index])
		#TS_Data=TS_Data.reshape([1])
		#print(TS_Data.shape)
		TS_Label=torch.from_numpy(np.array((self.Label[index])))
		#TS_Data=torch.from_numpy(TS_Data)
		if self.transform:
			TS_Data=self.transform(TS_Data)
		return (TS_Data,TS_Label)
