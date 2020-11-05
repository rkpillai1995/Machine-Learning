# -*- coding: utf-8 -*-
"""
Naive network to train and learn about the NSynth dataset.
This network is used for classification between the instrument class

@author: Gokul Govindarajan Chandraketu
@author: Rajkumar Lenin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
import common

class Net(nn.Module):
    """
    LeNet - 5 inspired network for NSynth classification
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        """
        Constructor which initializes the layers in the LeNet-5 inspired network
        """
        super(Net, self).__init__()
        
        # Hidden size for the LSTM network
        self.hidden_size = hidden_size
        
        # Stacked number of LSTMs
        self.num_layers = num_layers
        
        # Device in which the entire network is being run
        self.device = device
        
        # LSTM network and it's modal parameters
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Final layer
        self.fc1 = nn.Linear(hidden_size * 2, 10)

    def forward(self, x):
        """
        Forward method that'll move forward the with the given input across the Convolution Neural Network.
        :param x: inputs
        :return: the output value of the last layer
        """
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        
        x, _ = self.lstm(x, (h0, c0))
        
        # Applying final layer
        x = self.fc1(x[:, -1, :])

        # Softmax of the last layer. This way the probabilities are normalized
        x = F.softmax(x, dim=1)
        return x

def main():

    if "DISPLAY" not in os.environ:
        matplotlib.use("pdf")
    
    # This will pick the best device to run the network on currently
    deviceNumber = common.get_free_gpu()
    device = torch.device('cuda:' + str(deviceNumber) if torch.cuda.is_available() else 'cpu' )
    
    # To rerun the already trained network, running the network in the same device it was trained on.
    # This network was trained on cuda:1, hence I'm hardcoding it to cuda:1
    device = 'cuda:1'

    # Get the device to run the network on
    print(device)

    # Set the network parameters
    epochs = 30
    learning_rate = 0.0005
    input_size = 100
    hidden_size = 128
    num_layers = 2
    output_folder_name = "Q2"
    weights_file_name = "q2 weights.txt"

    # Set the base location of the dataset
    dataset_base_location = '/local/sandbox/nsynth/'
	
    # Create the network and load the network in the device
    net = Net(input_size, hidden_size, num_layers, device)
    net.to(device)

    # Run the RNN for the given parameters
    common.run_RNN(dataset_base_location, epochs, input_size, learning_rate, net, output_folder_name, device, weights_file_name)


if __name__ == '__main__':
    main()
