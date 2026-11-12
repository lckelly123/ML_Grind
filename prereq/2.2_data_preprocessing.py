import torch
import os
import pandas as pd

#make the csv file:
os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
print(os.path.abspath(data_file))
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

#load dataset:
data = pd.read_csv(data_file)
print(data)

#seperate into input and output
#get_dummies seperates the text categorical data into one-hot encoding
inputs,targets = data.iloc[:,0:2], data.iloc[:,2]
inputs = pd.get_dummies(inputs, dummy_na=True)

#converts Nans in numerical input cols to the col averages
inputs = inputs.fillna(inputs.mean())
print(inputs)

#convert to tensor:
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
