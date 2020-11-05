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

"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program creates a binary classifier with single perceptron
"""


# Using CUDA if we have a GPU that supports it along with the correct
# Install, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

EPOCHS = 20
MFCC_10_feature, MFCC_17_feauture, species= [], [], []

## Reading the files
filename=input("Please enter the filename(Frogs.csv/Frogs-subsample.csv): ")
with open(filename) as fi:
  datanumpyarray = np.loadtxt(fi, delimiter=",", dtype='str', comments="#", skiprows=1, usecols=(0, 1, 2))

### Storing the values of each feature in an array

for i in range(0, len(datanumpyarray)):
  MFCC_10_feature.append(float(datanumpyarray[i][0]))
  MFCC_17_feauture.append(float(datanumpyarray[i][1]))
  species.append(str(datanumpyarray[i][2]))



MFCC_10_feature = np.array(MFCC_10_feature)
MFCC_17_feauture = np.array(MFCC_17_feauture)




for i in range(0,len(species)):
    if species[i] == 'HylaMinuta':

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


  samples_x = MFCC_10_feature.astype(np.float)
  samples_y = MFCC_17_feauture.astype(np.float)
  for i in range(len(species)):
    if species[i] == 'HylaMinuta':
      x_1.append(samples_x[i])
      y_1.append(samples_y[i])
    else:
      x_2.append(samples_x[i])
      y_2.append(samples_y[i])

  return x_1, x_2, y_1, y_2











def plot_the_classifier(bi, w1, w2):
  '''

  :param bi: Bias
  :param w1: weight1
  :param w2: weight 2
  :return: Plots the line which represents the linear classification
  '''
  fig, ax = plt.subplots()

  x1, x2, y1, y2 = scatter_points()
  ax.scatter(x1, y1, marker="o", color='orange', label="Hylaminuta")

  ax.scatter(x2, y2, marker="x", color='blue', label="HypsiboasCinerascens")
  plt.legend()
  plt.xlabel('MFCC_10')
  plt.ylabel('MFCC_17')
  plt.title('Scatterplot with Linear classifier')

  slope = -(bi / w2) / (bi / w1)
  intercept = -bi / w2


  a = np.linspace(-0.5, 0.4, 100)
  b = slope * a + intercept
  plt.plot(a, b, '-r', label='y=2x+1')
  plt.show()










class Net(nn.Module):
    '''
    This class is used to define the neural net model
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)  # in features, out features

    def forward(self, z):
        z = self.fc1(z)
        z = torch.sigmoid(z)

        return z





if __name__=='__main__':


    userinput=input("Train the dataset(y/n): ")
    if userinput=="y":
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
        #print(net)

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

                running_loss += loss.item()
            print('Epoch ' + str(epoch + 1) + ', loss: ' + str(
                running_loss / 64))  # 50,000 = total number of images in training for
            # one epoch, this gets average loss per image
            # 781 * 64 = 49,984 (roughly the 50000 training images)
            running_loss = 0.0

        print('Finished Training')

        weights = list(net.parameters())

        temp1 = weights[0]
        temp2 = weights[1]

        initial_weights = (temp1.data).cpu().numpy()
        ## Get the weight1 ,weight2 of the model
        w1, w2 = initial_weights[0][0], initial_weights[0][1]

        bias = (temp2.data).cpu().numpy()
        ## Get the bias of the model
        bi = bias[0]
        plot_the_classifier(bi, w1, w2)

    else:
        ##### Plotting without training with the trained parameter values
        if filename =="Frogs.csv":
            bi,w1,w2=-0.08726013,-10.717289 ,-6.44434
            plot_the_classifier(bi, w1, w2)
        if filename == "Frogs-subsample.csv":
                bi, w1, w2 = -0.05523135 ,-1.8006793 ,-0.5903512
                plot_the_classifier(bi, w1, w2)









