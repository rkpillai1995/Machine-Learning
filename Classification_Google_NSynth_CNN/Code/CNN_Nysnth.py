# -*- coding: utf-8 -*-
"""
Naive network to train and learn about the NSynth dataset.
This network is used for classification between the instrument class

@author: Gokul Govindarajan Chandraketu
@author: Rajkumar Lenin
"""

import operator
import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import itertools
import os
import heapq


class Net(nn.Module):
    """
    LeNet - 5 inspired network for NSynth classification
    """

    def __init__(self):
        """
        Constructor which initializes the layers in the LeNet-5 inspired network
        """
        super(Net, self).__init__()

        # First Convolution layer in the CNN. This takes in the audio samples as input
        # and gives out 16 output audio samples as output. They kernal size is 
        # 1x5, as this is an audio sample.
        self.conv1 = nn.Conv1d(1, 4, 5)

        # Second Convolution layer in the CNN. This takes in 6 audio samples as 
        # input and gives out 16 audio samples as output. The kernel size is 
        # again 1x5.

        # Third layer in the CNN, which is a fully connected layer. It takes in 
        # the output of the previous network with 35952 nodes and outputs to 600
        # nodes in the next layer. It's just a linear layer and fully connected 
        # layer.
        self.fc1 = nn.Linear(7196, 600)
        
        
        # Fourth layer in the CNN, which is fully connected layer. It takes in an 
        # input of 600 nodes from the previous layer and gives an ouput of 
        # 120 nodes to next layer as output.
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        """
        Forward method that'll move forward the with the given input across the Convolution Neural Network.
        :param x: inputs
        :return: the output value of the last layer
        """

        # Input put through the first Convolution layer
        x = F.relu(self.conv1(x))

        # Max pooling the output of the first Convolution layer. This is a 
        # max-pool of kernal size 2. That way, the input audio signal will be 
        # reduced by half.
        x = F.max_pool1d(x, 5)

        # Flatting out all dimensions to just one dimension
        x = x.view(-1, self.num_flat_features(x))

        # Third layer in the CNN, which is fully connected
        x = F.relu(self.fc1(x))

        # Fourth later in the CNN, which is fully connected
        x = self.fc2(x)

        # Softmax of the last layer. This way the probabilities are normalized
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def read_dataset(datasetLocation):
    """
    Method to read the data sets from the given dataset location.

    :param datasetLocation: Base location where training, test and validation datasets are located
    :return: training loader, test loader and validation loader data sets
    """

    # Pre-processing transform to get the dataset in the range of [-1, 1]
    maxIntValue = np.iinfo(np.int16).max
    # toFloat = transforms.Lambda(lambda x: ((x / maxIntValue) + 1) / 2)
    toFloat = transforms.Lambda(lambda x: x / maxIntValue)

    # Reading the train dataset
    trainFolderName = "nsynth-train"
    train_dataset = NSynth(datasetLocation + trainFolderName, transform = toFloat,
        blacklist_pattern = ["synth_lead"],
        categorical_field_list = ["instrument_family", "instrument_source"])
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Read the test dataset
    testFolderName = "nsynth-test"
    test_dataset = NSynth(datasetLocation + testFolderName, transform=toFloat,
        blacklist_pattern = ["synth_lead"],
        categorical_field_list = ["instrument_family", "instrument_source"])
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Read the validation dataset
    validationFolderName = "nsynth-valid"
    validation_dataset = NSynth(datasetLocation + validationFolderName, transform=toFloat,
        blacklist_pattern = ["synth_lead"],
        categorical_field_list = ["instrument_family", "instrument_source"])
    validation_loader = data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

    # return the dataset loader
    return train_loader, test_loader, validation_loader


def get_dataset_size(dataset):
    """
    Method to get the size of the given dataset dataset
    :param dataset: Input Dataset
    :return: number of audio sample in the given dataset
    """
    total_count = 0
    for i, data in enumerate(dataset, 0):
        inputs, _, _, _ = data
        total_count += len(inputs)
    return total_count


def train_network(network, data_set, optimizer, criterion, start_ranges, end_ranges, device ="cpu"):
    """
    Train the given network for the given dataset, using the optimizer and the criterion
    :param network: Network which need to be trained
    :param data_set: Data set that need to be used for training
    :param optimizer: Optimizer
    :param criterion: Criterion
    :param start_ranges: List of start range for each sample, to sub-sample.
    :param end_ranges: List of end range for each sample, to sub-sample.
    :param device: Device in which the algorithm will be run
    :return: Overall loss value for the entire dataset
    """

    # Variable to accumulate the overall loss
    overall_loss = 0.0

    # Loop for each batch
    for i, data in enumerate(data_set, 0):

        # Get the inputs
        inputs, labels, instrument_source_target, targets = data
        
        # Sub sample based on the given range
        for index in range(len(start_ranges)):
            start_range = start_ranges[index]
            end_range = end_ranges[index]
            if index == 0:
                sub_sampled_dataset = inputs[:, start_range:end_range]
            else:
                sub_sampled_dataset = sub_sampled_dataset + inputs[:, start_range:end_range]

        # Expand the dimensions for ease of processing
        inputs = torch.tensor(np.expand_dims(sub_sampled_dataset, axis=1))

        # Moving the sub sampled array to GPU
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Current prediction of the network based on the current weights and bias
        outputs = network(inputs)

        # calculating the loss based on the given criterion function, For the predicted outputs
        loss = criterion(outputs, labels)

        # Learn based on the loss
        loss.backward()
        optimizer.step()

        # Accumulate the loss in this batch learning instance
        overall_loss += loss.item()

    # report the loss occurred in this training cycle
    return overall_loss


def predict_output(network, data_set, optimizer, criterion, start_ranges, end_ranges, device ="cpu"):
    """
    Predicts the output of the given network based on the given dataset
    :param network: network to use for prediction
    :param data_set: dataset to be predicted
    :param optimizer: optimizer
    :param criterion: criterion
    :param start_ranges: List of start range for each sample, to sub-sample.
    :param end_ranges: List of end range for each sample, to sub-sample.
    :param device: device to run the algorithm in
    :return: overall loss and total correct prediction count
    """

    # Overall loss for the entire prediction
    overall_loss = 0.0

    # total correct predictions for the entire dataset
    total_correct_predictions = 0

    # Iterate thru the dataset
    for i, data in enumerate(data_set, 0):
        # Get the inputs
        inputs, labels, instrument_source_target, targets = data

        # Sub sample based on the given range
        for index in range(len(start_ranges)):
            start_range = start_ranges[index]
            end_range = end_ranges[index]
            if index == 0:
                sub_sampled_dataset = inputs[:, start_range:end_range]
            else:
                sub_sampled_dataset = sub_sampled_dataset + inputs[:, start_range:end_range]

        # Expand the dimensions for ease of processing
        inputs = torch.tensor(np.expand_dims(sub_sampled_dataset, axis=1))

        # Moving the sub sampled array to GPU
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

        # Predict the output
        outputs = network(inputs)

        # Compute loss based on the prediction
        prediction_loss = criterion(outputs, labels)

        # class prediction
        _, predicted_class = torch.max(outputs.data, 1)

        # Counting the total number of correct prediction
        total_correct_predictions += (predicted_class == labels).sum().item()

        # Accumulate the loss in this batch learning instance
        overall_loss += prediction_loss.item()

    # Report the overall loss in this training cycle, and number of correct predictions
    return overall_loss, total_correct_predictions


def classify(network, data_set, start_ranges, end_ranges, device="cpu", location ="./"):
    """
    Classify the given dataset using the given network
    :param network: network which need to be used for classification
    :param data_set: dataset to classify
    :param start_ranges: List of start range for each sample, to sub-sample.
    :param end_ranges: List of end range for each sample, to sub-sample.
    :param device: device that should be used to run the algorithm
    :param location: Location where the results need to be saved
    :return:
    """

    # Variable to hold the total number of right prediction
    total_correct_predictions = 0

    # Correctness matrix which holds the number of successful classifications and mis-classification, for each class
    confusion_matrix = np.zeros((10, 10))

    # count of each class data
    class_counts = np.zeros(10)

    # variables to remember the highest probability of the correct classification
    # for each class, and it's corresponding audio sample
    correct_classification = [0] * 10
    correct_classification_audio_sample = [None] * 10

    # variables to remember the lowest probability of the wrong classification
    # for each class, and it's corresponding audio sample
    incorrect_classification = [1] * 10
    incorrect_classification_audio_sample = [None] * 10

    # variables to remember the higher probability of the correct classification
    # near the decision boundary for each class, and it's corresponding audio sample
    slightly_higher_classification = [0] * 10
    slightly_higher_classification_audio_sample = [None] * 10

    # variables to remember the lower probability of the wrong classification
    # near the decision boundary for each class, and it's corresponding audio sample
    slightly_lower_classification = [0] * 10
    slightly_lower_classification_audio_sample = [None] * 10

    # For each batch
    for i, data in enumerate(data_set, 0):

        # get input
        original_audio_samples, labels, instrument_source_target, targets = data

        # Sub sample based on the given range
        for index in range(len(start_ranges)):
            start_range = start_ranges[index]
            end_range = end_ranges[index]
            if index == 0:
                sub_sampled_dataset = original_audio_samples[:, start_range:end_range]
            else:
                sub_sampled_dataset = sub_sampled_dataset + original_audio_samples[:, start_range:end_range]

        # Expand the dimensions for ease of processing
        audio_samples = torch.tensor(np.expand_dims(sub_sampled_dataset, axis=1))

        # Moving the sub sampled array to GPU
        images_in_device, labels = audio_samples.to(device, dtype=torch.float), labels.to(device)

        # predict using the network
        prediction = network(images_in_device)
        cpu_prediction = prediction.data.to('cpu')

        # For each data sample in this batch
        for index in range(len(cpu_prediction)):

            # Probabilities for the current data sample
            prediction_probabilities = cpu_prediction[index].detach().numpy()

            # Get predicted class and the probability of the data sample being that class
            predicted_class, prediction_probability = max(enumerate(prediction_probabilities), key=operator.itemgetter(1))

            # Actual class of the current data sample
            actual_class = int(labels[index])

            # If the prediction is correct
            if actual_class == predicted_class:

                # Increment the total correct prediction
                total_correct_predictions += 1
                
                # Getting the data sample for each class which will be predicted with most certainty
                if correct_classification[actual_class] < prediction_probability:
                    correct_classification[actual_class] = prediction_probability
                    correct_classification_audio_sample[actual_class] = original_audio_samples[index]
                    
                # Getting the data sample for each class which will be predicted as the right class with least certainty.
                # Meaning, it just barely made it to the right class label
                second_max_probability = heapq.nlargest(2, prediction_probabilities)[1]
                if slightly_higher_classification[actual_class] < second_max_probability:
                    slightly_higher_classification[actual_class] = second_max_probability
                    second_max_class = sorted(range(len(prediction_probabilities)), key=lambda k: prediction_probabilities[k])[1]

                    # Record the predicted probability, missed probability and the class which we just avoided
                    slightly_higher_classification_audio_sample[actual_class] = [prediction_probability, second_max_probability, second_max_class, original_audio_samples[index]]

            # If the prediction is incorrect
            else:
                
                # Getting the data sample for each class which be predicted as a wrong class with most certainty.
                # Meaning, that it's very certain/confident in prediction and it went wrong 
                if incorrect_classification[actual_class] > prediction_probabilities[actual_class]:
                    incorrect_classification[actual_class] = prediction_probabilities[actual_class]

                    # Record the class that it predicted as with most certainty (and went wrong),
                    # and the audio sample for which this happened
                    incorrect_classification_audio_sample[actual_class] = [predicted_class, original_audio_samples[index]]
                
                # Getting the data sample for each class which will be predicted wrong with least uncertainity.
                # Meaning, it just barely didn't predict it right.
                if slightly_lower_classification[actual_class] < prediction_probabilities[actual_class]:
                    slightly_lower_classification[actual_class] = prediction_probabilities[actual_class]

                    # Record the probability of the class if predicted,
                    # actual class's prediction (which probably is slightly lesser than the right class),
                    # and the class label and the audio sample for which this happened
                    slightly_lower_classification_audio_sample[actual_class] = [prediction_probability, prediction_probabilities[actual_class], predicted_class, original_audio_samples[index]]
                    
            # Count the class
            class_counts[actual_class] += 1

            # Increment the confusion matrix
            confusion_matrix[actual_class][predicted_class] += 1

    # Save the predicted audio_samples
    plot_extreme_audio_samples(correct_classification_audio_sample, incorrect_classification_audio_sample, location)
    plot_border_audio_samples(slightly_higher_classification_audio_sample, slightly_lower_classification_audio_sample, location)

    # Return the values accordingly
    return total_correct_predictions, confusion_matrix, class_counts


def plot_extreme_audio_samples(correct_classifications, incorrect_classifications, location = "./"):
    """
    Saves the audio samples for most correct and most incorrect classification.
    Meaning, when the network classified with high certainty and was right,
    and when the network classified with high certainty and was wrong.

    :param correct_classifications: Correctly predicted images
    :param incorrect_classifications: Incorrectly predicted images
    :param location: Location where it need to be saved
    :return: None
    """

    # Save all correct predicted images
    index = -1
    for audio_sample in correct_classifications:
        index += 1

        # This means that this class was never classified correctly. Meaning, the success rate is 0%
        if audio_sample is None:
            continue

        # Plot and save the image
        fig, ax = plt.subplots()
        image_title = "Class - " + str(index) + " Predicted - " + str(index)
        image_name = location + "Class" + str(index) + "_Predicted" + str(index) + ".jpg"
        plt.title(image_title)
        plt.plot(audio_sample.numpy())
        fig.savefig(image_name)

    # Save all incorrectly predicted images
    index = -1
    for data in incorrect_classifications:
        index += 1
        if data is None:
            continue

        # Plot and save the image
        predicted_class, audio_sample = data[0], data[1]
        fig, ax = plt.subplots()
        image_title = "Class - " + str(index) + ", Predicted - " + str(predicted_class)
        image_name = location + "Class" + str(index) + "_Predicted" + str(predicted_class) + ".jpg"
        plt.title(image_title)
        plt.plot(audio_sample.numpy())
        fig.savefig(image_name)


def plot_border_audio_samples(slightly_higher_classification_data, slightly_lower_classification_data, location = "./"):
    """
    Saves the plots for audio samples which were near the decision boundary.
    Meaning it either got it right with a very slight margin,
    or it got it wrong with a very slight margin.

    In either cases, the network was quite uncertain.

    :param slightly_higher_classification_data: Data where the network got it right with slight margin. It was quite lucky
    :param slightly_lower_classification_data: Data where the network got it wrong with slight margin. It was quite unlucky
    :param location: Location where the images need to be saved
    :return: None
    """

    # For data where the network got it slightly right
    index = -1
    for correctClassification in slightly_higher_classification_data:
        index += 1
        if correctClassification is None:
            continue
        
        prediction_probability, second_max_probability, second_max_class, audio_sample = correctClassification[0], correctClassification[1], correctClassification[2], correctClassification[3]
        fig, ax = plt.subplots()
        image_title = "Class - " + str(index) + " (" + str(prediction_probability) + ") Missed Class - " + str(second_max_class) + " (" + str(second_max_probability) + ")"
        image_name = location + "JustGotIt_Class" + str(index) + "_Predicted" + str(index) + ".jpg"
        plt.title(image_title)
        plt.plot(audio_sample.numpy())
        fig.savefig(image_name)
        
    index = -1
    for incorrectClassification in slightly_lower_classification_data:
        index += 1
        if incorrectClassification is None:
            continue
        
        prediction_probability, actual_class_probability, predicted_class, audio_sample = incorrectClassification[0], incorrectClassification[1], incorrectClassification[2], incorrectClassification[3]
        fig, ax = plt.subplots()
        image_title = "Class - " + str(index) + " (" + str(actual_class_probability) + ") Predicted Class - " + str(predicted_class) + " (" + str(prediction_probability) + ")"
        image_name = location + "JustMissedIt_Class" + str(index) + "_Predicted" + str(predicted_class) + ".jpg"
        plt.title(image_title)
        plt.plot(audio_sample.numpy())
        fig.savefig(image_name)


def generate_line_graph(training_accuracy, test_accuracy, location ="./"):
    """
    Method to generate a line graph and save it for the given training and test accuracies
    :param training_accuracy: training accuracy for which the line graph need to be generated
    :param test_accuracy: test accuracy for which the line graph need to be generated
    :param location: location where the line graph image need to be stored
    :return: None
    """

    # Create the plot
    fig, ax = plt.subplots()
    imageName = "Learning rate"

    # Plotting the image
    ax.plot(range(1, len(training_accuracy) + 1), training_accuracy, label="Training Loss")
    ax.plot(range(1, len(training_accuracy) + 1), test_accuracy, label="Validation Loss")

    # Labeling the image and marking the axes
    plt.title(imageName)
    plt.xlabel("EPOCS")
    plt.ylabel("Cross-entropy")
    plt.legend()

    # Save the image
    fig.savefig(location + imageName + ".jpg")


def generate_histogram_class(confusion_matrix, class_counts, location ="./"):
    """
    Generate the histogram class for each class based on the confusion matrix given
    :param confusion_matrix: Confusion matrix for all classes
    :param class_counts: data sample counts for each class
    :param location: Location where the images need to be saved
    :return: None
    """

    # For each class
    for index in range(len(class_counts)):
        current_class_count = class_counts[index]

        # Get the percentage values for each class and prediction
        percents = []
        for index2 in range(len(class_counts)):
            percent = (confusion_matrix[index][index2] / current_class_count) * 100
            percents.append(percent)
            # print("%2.4f" % percent, end=' ')
        # print()

        image_title = "Classification of " + str(index) + " and it's accuracy"
        image_name = location + "Class_" + str(index) + "_Accuracy" + ".jpg"
        x_values = np.arange(len(class_counts))

        # Plotting the classification histogram for the current class
        fig, ax = plt.subplots()
        rects = plt.bar(x_values, percents, align = "center", alpha=0.5)
        plt.ylabel("Accuracy")
        plt.title(image_title)
        label_bar_chart(rects, percents, ax)
        fig.savefig(image_name)


def label_bar_chart(rects, percents, ax):
    for index in range(len(rects)):
        rect = rects[index]
        height = percents[index]
        ax.text(rect.get_x() + rect.get_width()/2., 1.01 * height, '%2.2f' % height, ha='center', va='bottom')


def create_confusion_matrix_image(confusion_matrix, class_counts, location ="./"):
    """
    Method to create the confusion matrix
    :param confusion_matrix: Confusion matrix
    :param class_counts: data sample counts for each class
    :param location: Location where the images need to be saved
    :return: None
    """

    # Normalized matrix based on the current class probability
    normalized_matrix = confusion_matrix
    for actualLabel in range(len(confusion_matrix)):
        for predictedLabel in range(len(confusion_matrix[0])):
            normalized_matrix[actualLabel, predictedLabel] = confusion_matrix[actualLabel, predictedLabel] / class_counts[actualLabel]

    # Plot the image
    fig, ax = plt.subplots()
    plt.imshow(normalized_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    fig.savefig(location + "ConfusionMatrix.jpg")


def main():

    # Get the device to run the network on
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set the base location of the dataset
    # dataset_base_location = "../Dataset/"
    dataset_base_location = '/local/sandbox/nsynth/'

    # Read the dataset
    train_loader, test_loader, validation_loader = read_dataset(dataset_base_location)

    # Create if the output folder doesn't exists.
    output_folder_name = "Q1"
    if not os.path.exists(output_folder_name):
        os.mkdir(output_folder_name)

    # Set the network parameters
    epochs = 15
    learning_rate = 0.01
    momentum = 0.3

    # Create the network and load the network in the device
    net = Net()
    net.to(device)

    # Load the network parameters if they exist
    weights_file_name = "q1 weights.txt"
    try:
        net.load_state_dict(torch.load(weights_file_name))
        epochs = -1
    except:
        print("weights file not found")

    # Create the criterion method
    criterion = nn.CrossEntropyLoss()

    # Create the optimizer to learn the mistakes of the network
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Training dataset information
    # train_dataset_size = getDatasetSize(train_loader)
    train_batch_count = len(train_loader)

    # Test dataset information
    test_dataset_size = get_dataset_size(test_loader)
    test_batch_count = len(test_loader)

    # Validation dataset information
    validation_dataset_size = get_dataset_size(validation_loader)
    validation_batch_count = len(validation_loader)

    # List to record training and validation loss after each epoch, so that it can be plotted later
    training_losses = []
    validation_losses = []

    # Record the classification accuracy of the validation dataset after each epoch
    classification_accuracies = []

    # Start and end range of the sub sampling
    start_range = [0]
    end_range = [9000]

    for epoch in range(epochs):

        # Train the training dataset
        train_loss = train_network(net, train_loader, optimizer, criterion, start_range, end_range, device)
        train_loss /= train_batch_count
        training_losses.append(train_loss)
        
        # train_loss = train_network(net, test_loader, optimizer, criterion, start_range, end_range, device)
        # train_loss /= test_batch_count
        # training_losses.append(train_loss)

        # Predict the output of the trained network for the validation dataset (after each epoch)
        validation_loss, correct = predict_output(net, validation_loader, optimizer, criterion, start_range, end_range, device)
        validation_loss /= validation_batch_count
        validation_losses.append(validation_loss)

        # Check how many of the validation dataset was classified properly and show the accuracy
        classification_accuracy = (correct * 100) / validation_dataset_size
        classification_accuracies.append(classification_accuracy)
        print('Classification accuracy in ' ,(epoch + 1) , ' epoch : %2.4f' % classification_accuracy)

    # Classify the network on the test dataset
    number_of_correct_classifications, confusion_matrix, class_counts = classify(net, test_loader, start_range, end_range, device, output_folder_name + "/")
    print("Test Data : " + str(number_of_correct_classifications) + " successfully classified out of " + str(test_dataset_size))

    # Generate the line graph of the training loss and the validation loss after each epoch
    generate_line_graph(training_losses, validation_losses, output_folder_name + "/")

    # Generate the histogram graph of each class in the test data set
    generate_histogram_class(confusion_matrix, class_counts, output_folder_name + "/")

    # Generate the confusion matrix image for the test dataset
    create_confusion_matrix_image(confusion_matrix, class_counts, output_folder_name + "/")

    # Save the network and it's parameters
    torch.save(net.state_dict(), output_folder_name + "/" + weights_file_name)
    # torch.save(net, output_folder_name + "/q1 network.pt")

if __name__ == '__main__':
    main()
