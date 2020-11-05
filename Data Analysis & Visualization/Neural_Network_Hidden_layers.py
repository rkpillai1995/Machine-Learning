__author__ = 'Rajkumar Pillai'

import torch
import torchvision
import numpy as np
import torch.nn as nn
import csv
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.ticker as ticker
from torchvision.datasets import MNIST
"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program creates create a neural network with single perceptron or hidden layers 
trained on Hastie-data.npy and shows different classification regions 
"""

# Using CUDA if we have a GPU that supports it along with the correct
# Install, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

EPOCHS = 30
MFCC_10_features, MFCC_17_features, species= [], [], []

## Reading the input file
datanumpyarray=np.load('Hastie-data.npy')


### Storing the values of each feature in an array

for i in range(0, len(datanumpyarray)):
  MFCC_10_features.append(float(datanumpyarray[i][0]))
  MFCC_17_features.append(float(datanumpyarray[i][1]))
  species.append(str(datanumpyarray[i][2]))



MFCC_10_features = np.array(MFCC_10_features)
MFCC_17_features = np.array(MFCC_17_features)




for i in range(0,len(species)):
    if species[i] == '0.0':

        datanumpyarray[i][2]= "0"
    else:

        datanumpyarray[i][2]= "1"

## To get an array of label

x = datanumpyarray[:, -1] # -1 means first from the end
label_array= x[np.newaxis, :].T
label_array=np.float64(label_array)

## To get an array of inputs

input_array =np.delete(datanumpyarray, -1, axis=1)
input_array=np.float64(input_array)


def scatter_points():
  '''
  This method returns the points for scatter plot
  :return: x1,x2,y1,y2 : The points that need to plotted using scatter plot
  '''
  x_1 = []
  x_2 = []
  y_1 = []
  y_2 = []


  samples_x = MFCC_10_features.astype(np.float)
  samples_y = MFCC_17_features.astype(np.float)
  for i in range(len(species)):
    if species[i]=='0.0':
      x_1.append(samples_x[i])
      y_1.append(samples_y[i])
    else:
      x_2.append(samples_x[i])
      y_2.append(samples_y[i])

  return x_1, x_2, y_1, y_2







def plot_the_classifier():
  '''

  :param bi: Bias
  :param w1: weight1
  :param w2: weight 2
  :return: Plots the line which represents the linear classification
  '''

  x1, x2, y1, y2 = scatter_points()
  ax.scatter(x1, y1, marker="o", color='orange', label="Hylaminuta")
  ax.scatter(x2, y2, marker="x", color='blue', label="HypsiboasCinerascens")
  plt.xlabel('MFCC_10')
  plt.ylabel('MFCC_17')
  plt.title('Scatterplot with class regions')
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)

  plt.show()








class HNet(nn.Module):
    '''
    This class is used to define the neural net model with Hidden Layers
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # in features, out features
        self.fc2 = nn.Linear(64,2)
        self.fc3 = nn.Linear(2,1)



    def forward(self, z):
        z = self.fc1(z)

        z = self.fc2(z)
        z = torch.sigmoid(z)

        z =  self.fc3(z)
        z = torch.sigmoid(z)


        return z


#    This class is used to define the neural net model with single perceptron
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(2, 1)  # in features, out features
#
#     def forward(self, z):
#         z = self.fc1(z)
#         z = torch.sigmoid(z)
#
#         return z

class Net(nn.Module):
    '''
    This class is used to define the neural net model with single perceptron

    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # in features, out features
        self.fc2 = nn.Linear(64,2)
        self.fc3 = nn.Linear(2,1)



    def forward(self, z):
        z = self.fc1(z)

        z = self.fc2(z)
        z = torch.sigmoid(z)

        z =  self.fc3(z)
        z = torch.sigmoid(z)


        return z





if __name__=='__main__':


    userinput=input("Train the dataset(y/n)")
    if userinput=="y":
        ############################# Training ##################

        trainset = torch.utils.data.TensorDataset(torch.from_numpy(input_array).float(),
                                                  torch.from_numpy(label_array).float())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=2)


        classes = ('HylaMinuta', 'HypsiboasCinerascens')

        ####################################
        net = Net()
        net.to(device)



        criterion = torch.nn.MSELoss(reduction='sum')

        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        ###########################################


        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # Get the inputs

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = net(inputs)


                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()


            print('Epoch ' + str(epoch + 1))  # 50,000 = total number of images in training for
            # one epoch, this gets average loss per image
            # 781 * 64 = 49,984 (roughly the 50000 training images)

        print('Finished Training')
        weights = list(net.parameters())


        ###################### Plotting all points with correspong class region color
        x = np.linspace(-3, 5, 100)
        label_array = np.linspace(-3, 5, 100)
        coordinates=[]
        x_blue=[]
        x_orange=[]

        y_blue = []
        y_orange = []
        for items in x:
            for item in label_array:
                coordinates.append([items, item])
                inputs = torch.Tensor(torch.from_numpy(np.float64([items,item])).float())
                inputs = inputs.to(device)
                outputs = net(inputs)
                ansOfOutput = ((outputs.data).cpu().numpy())
                if ansOfOutput[0]>0.5:
                    x_blue.append(items)
                    y_blue.append(item)

                else:
                    x_orange.append(items)
                    y_orange.append(item)


        fig, ax = plt.subplots()
        ax.scatter(x_blue, y_blue, color='red', label="HypsiboasCinerascens-class-region")
        ax.scatter(x_orange, y_orange, color='yellow', label="HylaMinuta-class-region")
        plot_the_classifier()
    else:


        trainset = torch.utils.data.TensorDataset(torch.from_numpy(input_array).float(),
                                                  torch.from_numpy(label_array).float())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=2)

        ####################################
        net = Net()
        net.to(device)



        criterion = torch.nn.MSELoss(reduction='sum')

        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        ###########################################

        for epoch in range(EPOCHS):
            for i, data in enumerate(trainloader, 0):
                # Get the inputs

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = net(inputs)


                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()



        weights = list(net.parameters())
        ###################### Plotting all points with correspong class region color

        x = np.linspace(-3, 5, 100)
        label_array = np.linspace(-3, 5, 100)
        coordinates=[]
        x_blue=[]
        x_orange=[]

        y_blue = []
        y_orange = []
        for items in x:
            for item in label_array:
                coordinates.append([items, item])
                inputs = torch.Tensor(torch.from_numpy(np.float64([items,item])).float())
                inputs = inputs.to(device)
                outputs = net(inputs)
                ansOfOutput = ((outputs.data).cpu().numpy())
                if ansOfOutput[0]>0.5:
                    x_blue.append(items)
                    y_blue.append(item)

                else:
                    x_orange.append(items)
                    y_orange.append(item)


        fig, ax = plt.subplots()
        ax.scatter(x_blue, y_blue, color='red', label="HypsiboasCinerascens-class-region")
        ax.scatter(x_orange, y_orange, color='yellow', label="HylaMinuta-class-region")
        plot_the_classifier()
