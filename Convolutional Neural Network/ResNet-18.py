__author__ = 'Rajkumar Pillai'

import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
import torch.optim as optim

import time


"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program is used to test Resnet18 model on FashionMNIST dataset

"""

# Using CUDA if we have a GPU that supports it along with the correct
# Install, otherwise use the CPU

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

##Declaring the no of epochs
EPOCHS = 10
train_set_sampler_class = []
val_set_sampler_class = []

######## Used to store the samples belonging to specific class
class_zero_dataset = []
class_one_dataset = []
class_two_dataset = []
class_three_dataset = []
class_four_dataset = []
class_five_dataset = []
class_six_dataset = []
class_seven_dataset = []
class_eight_dataset = []
class_nine_dataset = []

if __name__ == '__main__':

    ### Transformomg the data in order to use resnet18 model
    data_transform = transforms.Compose(
        [
         transforms.Resize((224, 224)),    ### Resizing image dimenesion to 224x224
         transforms.ToTensor(),
         transforms.Lambda(lambda x: torch.squeeze(x)),  ### To remove dimeensions of input of size 1
         transforms.Lambda(lambda x: torch.stack([x, x, x], 0)),  ### To convert the image to RGB by replicating in each band

        ])


    dataset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=True,
                                         download=True, transform=data_transform)

    # Loading the training set
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                              num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=False,

                                         download=True, transform=data_transform)
    #print(testset)

    #for i in range (len(testset)):
    #  print(testset[i][1])
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                              shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')



    net=models.resnet18(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc=nn.Linear(512,10)
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.2)

    #####################################
    ## Plotting the accuracies
    print("Accuracy calculation")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the  test images: %d %%\n' % (
        100 * correct / total))

    ##Used to print the  accuracies of correct samples
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    ##Used to print the  accuracies of incorrect samples
    class_incorrect = list(0. for i in range(10))
    class_incorrect_total = list(0. for i in range(10))

    ##Used to store the accuracies for each class which is used for plotting
    class_zero_plot = [0] * 10
    class_one_plot = [0] * 10
    class_two_plot = [0] * 10
    class_three_plot = [0] * 10
    class_four_plot = [0] * 10
    class_five_plot = [0] * 10
    class_six_plot = [0] * 10
    class_seven_plot = [0] * 10
    class_eight_plot = [0] * 10
    class_nine_plot = [0] * 10

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            c_incorrect = (predicted != labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

                class_incorrect[label] += c_incorrect[i].item()
                class_incorrect_total[label] += 1

            ### Used to compute the array which can be used to plot accuracies of each individual class
            labels = np.asarray(labels)
            predicted = np.asarray(predicted)
            for i in range(len(labels)):
                if labels[i] == 0:
                    class_zero_plot[predicted[i]] += 1
                if labels[i] == 1:
                    class_one_plot[predicted[i]] += 1
                if labels[i] == 2:
                    class_two_plot[predicted[i]] += 1
                if labels[i] == 3:
                    class_three_plot[predicted[i]] += 1
                if labels[i] == 4:
                    class_four_plot[predicted[i]] += 1
                if labels[i] == 5:
                    class_five_plot[predicted[i]] += 1
                if labels[i] == 6:
                    class_six_plot[predicted[i]] += 1
                if labels[i] == 7:
                    class_seven_plot[predicted[i]] += 1
                if labels[i] == 8:
                    class_eight_plot[predicted[i]] += 1
                if labels[i] == 9:
                    class_nine_plot[predicted[i]] += 1

    ### To print accuracy of each class
    accuracies = []
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]), " Incorrect samples ", int(class_incorrect[i]),
              " correct samples ", int(class_correct[i]))

        ### Storing the accuracies of all classes
        accuracies.append((class_correct[i] / class_total[i]) * 100)


    def train(trainloader):
        '''
        This function is used to Train the network
        :param trainloader:   The 80% training set
        :param validationloader: The 20% validation set
        :return: saved_losses_train   The losses in training of network
        :return:saved_losses_validation The errors during validation
        :return:overfitting_epoch_train  The epoch from which overfitting occurs in training set
        :return:overfitting_epoch_val    The epoch from which overfitting occurs in validation set
        '''

        print("Training started")
        saved_losses_train = []
        saved_losses_validation = []
        prev_loss_val = 0
        flag_val = False
        overfitting_epoch_train = 0
        overfitting_epoch_val = 0  # To store the epoch at which overfitting occurs
        start = time.time()

        for epoch in range(EPOCHS):
            running_loss = 0.0
            Val_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # Get the inputs
                inputs, labels = data
                # print(inputs)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimizer
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                running_loss += loss.item()

                optimizer.step()

            # Print statistics
            saved_losses_train.append(running_loss / 750)
            print('Epoch ' + str(epoch + 1) + ', Train_loss: ' + str(
                running_loss / 750))



            prev_loss_val = Val_loss

        print('Finished Training')
        end = time.time()

        ## Printing the time required to train the network
        print("Time to converge", end - start)

        return saved_losses_train, saved_losses_validation, overfitting_epoch_train, overfitting_epoch_val


    train(trainloader)


    ## Plotting the accuracies
    print("Accuracy calculation")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the  test images: %d %%\n' % (
        100 * correct / total))

    ##Used to print the  accuracies of correct samples
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    ##Used to print the  accuracies of incorrect samples
    class_incorrect = list(0. for i in range(10))
    class_incorrect_total = list(0. for i in range(10))

    ##Used to store the accuracies for each class which is used for plotting
    class_zero_plot = [0] * 10
    class_one_plot = [0] * 10
    class_two_plot = [0] * 10
    class_three_plot = [0] * 10
    class_four_plot = [0] * 10
    class_five_plot = [0] * 10
    class_six_plot = [0] * 10
    class_seven_plot = [0] * 10
    class_eight_plot = [0] * 10
    class_nine_plot = [0] * 10

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            c_incorrect = (predicted != labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

                class_incorrect[label] += c_incorrect[i].item()
                class_incorrect_total[label] += 1

            ### Used to compute the array which can be used to plot accuracies of each individual class
            labels = np.asarray(labels)
            predicted = np.asarray(predicted)
            for i in range(len(labels)):
                if labels[i] == 0:
                    class_zero_plot[predicted[i]] += 1
                if labels[i] == 1:
                    class_one_plot[predicted[i]] += 1
                if labels[i] == 2:
                    class_two_plot[predicted[i]] += 1
                if labels[i] == 3:
                    class_three_plot[predicted[i]] += 1
                if labels[i] == 4:
                    class_four_plot[predicted[i]] += 1
                if labels[i] == 5:
                    class_five_plot[predicted[i]] += 1
                if labels[i] == 6:
                    class_six_plot[predicted[i]] += 1
                if labels[i] == 7:
                    class_seven_plot[predicted[i]] += 1
                if labels[i] == 8:
                    class_eight_plot[predicted[i]] += 1
                if labels[i] == 9:
                    class_nine_plot[predicted[i]] += 1

    ### To print accuracy of each class
    accuracies = []
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]), " Incorrect samples ", int(class_incorrect[i]),
              " correct samples ", int(class_correct[i]))

        ### Storing the accuracies of all classes
        accuracies.append((class_correct[i] / class_total[i]) * 100)



    ## Saving the model weights
    torch.save(net.state_dict(), 'q3_model_weights')


