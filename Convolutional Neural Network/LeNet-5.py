__author__ = 'Rajkumar Pillai'

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import time


"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program is used to test the nearly LeNet5 model on Fashion-MNIST test set and printing the accuracies and then applying transfer learning
to adapt to Fashion-MNIST and to print and plot the learning curve and the accuracies for each class.

"""

# Using CUDA if we have a GPU that supports it along with the correct
# Install, otherwise use the CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

##Declaring the no of epochs

EPOCHS = 20
train_set_sampler_class = []   # Used to store the index of 80% training samples
val_set_sampler_class = []     # Used to store the index of 20% validation samples

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




    # Defining the network
    class Net(nn.Module):
        '''
        This class is used to define the nueral network according to the specifications  mentioned in question
        '''
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 70, 3)  # in channles, out channels, kernel size

            self.conv2 = nn.Conv2d(70, 480, 3)

            self.pool = nn.MaxPool2d(3)  # kernel size and stride

            self.fc1 = nn.Linear(1920, 20)

        def forward(self, x):
            x_size = x.size(0)
            x = self.pool(F.relu(self.conv1(x)))

            x = self.pool(F.relu(self.conv2(x)))

            x = x.view(x_size, -1)

            x = self.fc1(x)
            x = F.log_softmax(x)
            return x

    #####################################################################################################

    ##Downloading the FashionMNIST dataset


    dataset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=True,
                                         download=True, transform=transforms.ToTensor())


    testset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=False,

                                         download=True, transform=transforms.ToTensor())


    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                              shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')



    net = Net()
    net.load_state_dict(torch.load('q1_model_weights'))
    net.to(device)




    #Accuracy calculation Before training
    print("Accuracy calculation before transfer learning")

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

    print('\nAccuracy of the nearly LeNet5 network on the FashionMNIST test images: %d %%\n' % (
        100 * correct / total))

    ##Used to print the  accuracies of correct samples
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    ##Used to print the  accuracies of incorrect samples
    class_incorrect = list(0. for i in range(10))
    class_incorrect_total = list(0. for i in range(10))

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

    ### To print accuracy of each class using nearly LeNet5 model
    accuracy_cal = []
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]), " Incorrect samples ", int(class_incorrect[i]),
              " correct samples ", int(class_correct[i]))
        accuracy_cal.append((class_correct[i] / class_total[i]) * 100)


    #### Seperating samples for every class and string their index in array
    print("Seperating as per class...Please wait")
    for i in range(0, len(dataset)):
        # continue
        if dataset[i][1] == 0:
            class_zero_dataset.append(i)
        if dataset[i][1] == 1:
            class_one_dataset.append(i)
        if dataset[i][1] == 2:
            class_two_dataset.append(i)
        if dataset[i][1] == 3:
            class_three_dataset.append(i)
        if dataset[i][1] == 4:
            class_four_dataset.append(i)
        if dataset[i][1] == 5:
            class_five_dataset.append(i)
        if dataset[i][1] == 6:
            class_six_dataset.append(i)
        if dataset[i][1] == 7:
            class_seven_dataset.append(i)
        if dataset[i][1] == 8:
            class_eight_dataset.append(i)
        if dataset[i][1] == 9:
            class_nine_dataset.append(i)
    print("Seperation of class done")



    ## Splitting samples of each class into 80% training and 20% validation split and stroing their index in array
    def splittingLogic(dataset, train_set_sampler_class, val_set_sampler_class):
        '''
        This is used to split the samples for each class into 80%training and 20%testing and storing their index
        :param dataset: The entire dataset
        :param train_set_sampler_class:  To store the index of 80% training of each individual class
        :param val_set_sampler_class:    To store the index of 20% validation of each individual class
        '''
        X_train, X_val = train_test_split(dataset, test_size=0.2)
        # print(X_train)
        train_set_sampler_class += X_train
        val_set_sampler_class += X_val

    ## Calling the above function to split data for each class
    splittingLogic(class_zero_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_one_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_two_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_three_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_four_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_five_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_six_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_seven_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_eight_dataset, train_set_sampler_class, val_set_sampler_class)
    splittingLogic(class_nine_dataset, train_set_sampler_class, val_set_sampler_class)

    #Storing the samples of each class as per the index obtained from previous operations
    train_set_sampler = SubsetRandomSampler(train_set_sampler_class)
    validation_set_sampler = SubsetRandomSampler(val_set_sampler_class)

    # Loading the training set
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                              num_workers=2, sampler=train_set_sampler)



    #Loading the validation set
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                   num_workers=2, sampler=validation_set_sampler)
    print("Loading done")


    ###############################################
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.2)

    ##############################################


    def train(trainloader, validationloader):
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

                if (prev_loss_val >= Val_loss):  ## Checking the validation error compared to previous validation error
                    optimizer.step()

            # Print statistics
            saved_losses_train.append(running_loss / 750)
            print('Epoch ' + str(epoch + 1) + ', Train_loss: ' + str(
                running_loss / 750))

            Val_loss = 0.0
            ## Calculating the validation error
            for i, data in enumerate(validationloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                Val_loss += loss.item()
            saved_losses_validation.append(Val_loss / 750)
            print('Epoch ' + str(epoch + 1) + ', Val_loss: ' + str(
                Val_loss / 750))
            if (prev_loss_val < Val_loss and prev_loss_val != 0 and flag_val == False):
                print("Prev loss_validation ", prev_loss_val / 750)
                print("Runnning loss_validation ", Val_loss / 750)
                breakpoint = prev_loss_val
                overfitting_epoch_val += epoch + 1
                break

            prev_loss_val = Val_loss

        print('Finished Training')
        end = time.time()

        ## Printing the time required to train the network
        print("Time to converge", end - start)

        return saved_losses_train, saved_losses_validation, overfitting_epoch_train, overfitting_epoch_val

    def ploting(saved_losses_train,saved_losses_validation, overfitting_epoch_train,overfitting_epoch_val):

        '''
        To plot the  a learning curve plot showing the cross-entropy vs. epoch for both the training set and the validation set
        :param saved_losses_train:       The losses in training of network
        :param saved_losses_validation:  The errors during validation
        :return:overfitting_epoch_train  The epoch from which overfitting occurs in training set
        :return:overfitting_epoch_val    The epoch from which overfitting occurs in validation set
        '''

        print("Plotting")
        fig, ax = plt.subplots()

        x = np.linspace(1, len(saved_losses_train), len(saved_losses_validation))
        saved_losses_train = np.array(saved_losses_train)
        saved_losses_validation = np.array(saved_losses_validation)
        ax.set_title("Average Model Loss over Epochs")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Average Loss")

        ax.set_xticks(np.arange(len(saved_losses_validation) + 1))
        ax.plot(x, saved_losses_train, color='purple', marker=".", label="Train_set_loss")
        plt.legend()

        plt.axvline(overfitting_epoch_val, color='red', label="overfitting_epoch")
        plt.legend()

        ax.plot(x, saved_losses_validation, color='orange', marker=".", label="Validation_set_loss")
        plt.legend()

        fig.savefig('model_loss')
        plt.legend()
        plt.show(fig)


        ## Calling the function train to start training the network


    saved_losses_train, saved_losses_validation, overfitting_epoch_train, overfitting_epoch_val = train(trainloader,
                                                                                                        validationloader)

    ## To plot the learning curve
    ploting(saved_losses_train, saved_losses_validation, overfitting_epoch_train, overfitting_epoch_val)

    #####################################################################################################

    ## Saving the model weights
    torch.save(net.state_dict(), 'q2_model_weights')



    try:
        print("Accuracy calculation after training")
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

        dataiter = iter(testloader)
        images, labels = dataiter.next()

        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        # To see the actual values for each of the classes uncomment this
        # print(outputs)

        _, predicted = torch.max(outputs, 1)

        ### To plot the accuracies across all classes and each individual class
        print("Accuracies Plotting")

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        print("Plotting Accuracies of all classes")
        plt.bar(classes, accuracies)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Accuracies of all classes')
        plt.show()

        print('Plotting of Accuracy of class 0')
        plt.bar(classes, class_zero_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 0')
        plt.show()

        print('Plotting of Accuracy of class 1')
        plt.bar(classes, class_one_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 1')
        plt.show()

        print('Plotting of Accuracy of class 2')
        plt.bar(classes, class_two_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 2')
        plt.show()

        print('Plotting of Accuracy of class 3')
        plt.bar(classes, class_three_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 3')
        plt.show()

        print('Plotting of Accuracy of class 4')
        plt.bar(classes, class_four_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 4')
        plt.show()

        print('Plotting of Accuracy of class 5')
        plt.bar(classes, class_five_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 5')
        plt.show()

        print('Plotting of Accuracy of class 6')
        plt.bar(classes, class_six_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 6')
        plt.show()

        print('Plotting of Accuracy of class 7')
        plt.bar(classes, class_seven_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 7')
        plt.show()

        print('Plotting of Accuracy of class 8')
        plt.bar(classes, class_eight_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 8')
        plt.show()

        print('Plotting of Accuracy of class 9')
        plt.bar(classes, class_nine_plot)
        plt.xlabel("classes")
        plt.ylabel("accuracies")
        plt.title('Plot of Accuracy of class 9')
        plt.show()




    except:
        print()
