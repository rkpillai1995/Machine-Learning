__author__ = 'Rajkumar Pillai'

import numpy as np
import math
"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program creates naive bayes classifier according to the type specified by user 
 and  dataset from q3.csv file is used to tain the model  and performs classification of dataset for values of features in q3b.csv file
"""
def main():
    '''
    The main function takes input from user for thr type of classifier and performs classification and shows the
    accuracy and error rate of the classifier
    :return:
    '''

    ## Loading the data set from files with different columns according to requirement for training and testing the model
    file_name ='q3.csv'
    with open(file_name) as f:
      dataset = np.loadtxt(f, delimiter=",", dtype='str', comments="#",  skiprows=(1),usecols=(0,1,2,3,4,5,6,7,8))

    file_name = 'q3b.csv'
    with open(file_name) as f1:
        numerictestSet = np.loadtxt(f1, delimiter=",", dtype='str', comments="#", skiprows=(1),
                             usecols=( 6, 7))

    file_name = 'q3b.csv'
    with open(file_name) as f2:
        testSet = np.loadtxt(f2, delimiter=",", dtype='str', comments="#", skiprows=(1),
                                     usecols=(6, 7, 8))

    file_name = 'q3.csv'
    with open(file_name) as f3:
        datasetcategory = np.loadtxt(f3, delimiter=",", dtype='str', comments="#", skiprows=(1), usecols=(0, 1, 2, 3, 4, 5, 8))

    file_name = 'q3b.csv'
    with open(file_name) as f4:
        testSetcategorical = np.loadtxt(f4, delimiter=",", dtype='str', comments="#", skiprows=(1),
                             usecols=(0, 1, 2, 3, 4, 5))


    ### Calculating the MLE for Numeric attributes
    mean_and_std_of_attributes_by_class_numeric=mean_std_of_attributesnumeric(dataset)
    print("MLE of the Numeric attributes")
    for key,value in mean_and_std_of_attributes_by_class_numeric.items():
        print(key,value)
    print("######################################################################################################")


    ### Calculating the MLE for Categoric attributes
    print("MLE of the Categoric attributes")
    Likelihood_for_Category = CalcualteTrueFalseProbabilityCategory(datasetcategory)
    for key,value in Likelihood_for_Category.items():
        print(key,value)
    print("#######################################################################################################")

    ## Typecasting the input data as numpy array of float
    numerictestSet=numerictestSet.astype(np.float)

    option=input("Classification of data sample with full feature set (y/n)")
    if option=="y":

     ## The predict function for all features in the dataset
     labels=Predictionofdataset(numerictestSet, mean_and_std_of_attributes_by_class_numeric, Likelihood_for_Category,testSetcategorical)

    if option=="n":
     option = input("Classification of data sample with numeric feature or categoric feature set (1/2)")
     if option == "1":

         ## The predict function for all numeric in the dataset
         labels=Predictionofdatasetnumeric(numerictestSet, mean_and_std_of_attributes_by_class_numeric)

     if option == "2":

         ## The predict function for all categoric in the dataset
         labels=Predictionofdatasetcategory( Likelihood_for_Category,testSetcategorical)




    ## To calculate the error and accuracy of model
    accuracycalculation(testSet,labels)







def Predictionofdataset(numerictestSet,mean_and_std_of_attributes_by_class_numeric,Likelihood_for_Category,testSetcategorical):
    '''
    This function performs classification of data with all the features
    :param numerictestSet: The test dataset with the values of numeric attributes
    :param mean_and_std_of_attributes_by_class_numeric: mean and standadrd deviation of each numeric attribute for each class
    :param Likelihood_for_Category: The likelihood for each categoric attributes
    :param testSetcategorical: The test dataset with the values of categoric attributes
    :return: labels : the labels to be predicted
    '''
    labels=[]
    for i in range(len(numerictestSet)):

      ## Calculating the likelihood probabilites for numeric and categoric features
      probabilities_of_each_class_numeric = calprobability(mean_and_std_of_attributes_by_class_numeric, numerictestSet[i])
      probabilities_category = probabilitys(Likelihood_for_Category, testSetcategorical[i])
      TotalLikelihoodforFalse=probabilities_of_each_class_numeric['False']*probabilities_category['False']
      TotalLikelihoodforTrue=probabilities_of_each_class_numeric['True']*probabilities_category['True']

      if TotalLikelihoodforTrue > 0.5:
          label="True"
      else:
          label="False"
      labels.append(label)

    return labels


def Predictionofdatasetcategory(Likelihood_for_Category,testSetcategorical):
    '''
    This function performs classification of data with categoric features
    :param Likelihood_for_Category: The likelihood for each categoric attributes
    :param testSetcategorical: The test dataset with the values of categoric attributes
    :return: labels : the labels to be predicted
    '''
    labels=[]
    for i in range(len(testSetcategorical)):

     ## Calculating the likelihood probabilites for categoric features
     probabilities_category = probabilitys(Likelihood_for_Category, testSetcategorical[i])

     ## Predicting the labels
     currentLabel=""
     currentProbability = 0.5
     for class_value, probability in probabilities_category.items():
          if currentLabel=="" or probability > currentProbability:
              currentProbability = probability
              currentLabel = class_value

     labels.append(currentLabel)
    return labels



def Predictionofdatasetnumeric(numerictestSet,mean_and_std_of_attributes_by_class_numeric):
    '''
    This function performs classification of data with numeric features
    :param numerictestSet:     The test dataset with the values of numeric attributes
    :param mean_and_std_of_attributes_by_class_numeric: The likelihood for numeric attributes
    :return: labels : the labels to be predicted
    '''

    labels=[]
    for i in range(len(numerictestSet)):

      ## Calculating the likelihood probabilites for numeric features
      probabilities_of_each_class_numeric = calprobability(mean_and_std_of_attributes_by_class_numeric, numerictestSet[i])

      ## Predicting the labels
      currentLabel=""
      currentProbability = 0.5
      for class_value, probability in probabilities_of_each_class_numeric.items():
          if currentLabel=="" or probability > currentProbability:
              currentProbability = probability
              currentLabel = class_value

      labels.append(currentLabel)
    return labels


def accuracycalculation(testSet,labels):
    '''
    This function performs calculation of accuracy and classification error rate
    :param testSet: The test set which has the real labels
    :param labels: The set which has the predicted labels
     '''
    correctly_classified = 0
    i=0
    while i<len(testSet):

        if testSet[i][-1] == labels[i]:
                correctly_classified += 1
        i=i+1
    result= (correctly_classified / float(len(testSet)))
    accuracy=result*100.0
    error=100.0-accuracy
    print("Accuracy is: ",accuracy,"%")
    print("Classification-error is: ",error,'%')
    


def calprobability(mean_and_std_of_attributes_by_class,attributeValuesoftestset):
    '''
    This function calcualtes probabilities for numeric attribute values
    :param mean_and_std_of_attributes_by_class: The likelihood for numeric attributes
    :param attributeValuesoftestset: Values of attributes
    :return: probabilities_of_each_class : Calcualted probabilities for numeric attribute values
    '''
    probabilities_of_each_class = {}
    for class_value, mean_and_std_list in mean_and_std_of_attributes_by_class.items():
        probabilities_of_each_class[class_value] = 1
        i=0
        while i< len(mean_and_std_list):
                mean, stdev = mean_and_std_list[i]
                attribute_value = attributeValuesoftestset[i]
                probabilities_of_each_class[class_value] *= calc_likelihoodProbability(attribute_value, mean, stdev)
                i=i+1
    return probabilities_of_each_class






def calc_likelihoodProbability(attribute_value,mean,standarddeviation):
    '''
    This function performs the likehood probability calculation for each numeric attribute values
    according to the formula mentioned in writeup
    :param attribute_value: Values of attribute
    :param mean: mean of the attribute
    :param standarddeviation: Standard deviation of attribute
    :return: result: likehood probability  for each numeric attribute value
    '''

    result=(math.exp(-(math.pow(attribute_value-mean,2)/(2*math.pow(standarddeviation,2)))))* (1 / (math.sqrt(2*math.pi) * standarddeviation))
    return result

def dataseperationByClassforcategorical(dataset):
    '''
    Seperating the categoical attribute'stuples for each class Value
    :param dataset: the entire dataset
    :return: tuplesofeachclass: tuples for each class Value
    '''
    tuplesofeachclass = {}
    i=0
    while i<len(dataset):
        temp = dataset[i]
        if (temp[6] not in tuplesofeachclass):
            tuplesofeachclass[temp[6]] = []
        tuplesofeachclass[temp[6]].append(temp)
        i=i+1

    return tuplesofeachclass


def datasetseperationbyclass(dataset):
    '''
    Seperating the numeric attribute's tuples for each class Value
    :param dataset: the entire dataset
    :return: tuplesofeachclass: tuples for each class Value
    '''
    tuplesofeachclass = {}
    i=0
    while i<len(dataset):
        temp = dataset[i]
        if (temp[8] not in tuplesofeachclass):
            tuplesofeachclass[temp[8]] = []
        tuplesofeachclass[temp[8]].append(temp)
        i=i+1

    return tuplesofeachclass

def mean(ValuesofAttributes):
    '''
    This function calculates the mean of a feature
    :param ValuesofAttributes: Attribute's value
    :return: meanValue: The mean of the attribute
    '''
    ValuesofAttributes=ValuesofAttributes.astype(np.float)
    meanValue = np.mean(ValuesofAttributes)
    return meanValue


def standardDeviation(ValuesofAttributes):
    '''
    This function calculates the mean of a feature

    :param ValuesofAttributes: Attribute's value
    :return:  standardDeviationValue: The standardDeviation of the attribute
    '''
    ValuesofAttributes=ValuesofAttributes.astype(np.float)
    standardDeviationValue =np.std(ValuesofAttributes,ddof=1)
    return standardDeviationValue

def calc_mean_std(dataset):
    '''
    This function stores the mean standard deviation for each attribute
    :param dataset: The entire dataset
    :return: list_of_mean_std : Contains the mean standard deviation for each attribute
    '''
    attribute=dataset[:, 6]
    list_of_mean_std=[]
    list_of_mean_std.append((mean(attribute),standardDeviation(attribute)))

    attribute = dataset[:, 7]
    list_of_mean_std.append((mean(attribute), standardDeviation(attribute)))

    return list_of_mean_std


def mean_std_of_attributesnumeric(dataset):
    '''
    This function stores the mean standard deviation for each class
    :param dataset: The entire dataset
    :return: mean_and_std_of_attributes_by_class : Contains the mean standard deviation for each class
    '''
    dataseperatedbyclass=datasetseperationbyclass(dataset)
    mean_and_std_of_attributes_by_class={}
    for key,value in dataseperatedbyclass.items():
        value=np.array(value)
        mean_and_std_of_attributes_by_class[key]=calc_mean_std(value)

    return mean_and_std_of_attributes_by_class



def probabilitys(likelihood,testset):
    '''
    This function calculates the likelihood probabilites for the categoric attribute
    :param likelihood: The likelihood of the categoric attribute
    :param testset: the test set with categoric attributes
    :return: probabilities: likelihood probabilites of the categoric attribute
    '''
    probabilities = {}
    for class_value, values in likelihood.items():
        probabilities[class_value] = 1
        for i in range(len(values)):
            FalseProbab,TrueProbab = values[i]
            if testset[i]=="False":

                probabilities[class_value]*=FalseProbab
            if testset[i] == "True":

                probabilities[class_value] *= TrueProbab

    return probabilities

def CalcualteTrueFalseProbabilityCategory(dataset):
    '''
    This function calls other function to perform the calculation for likelihood prbability
    :param dataset: The entire dataset
    :return: TrueFalseProbability :The dictionary with probabilites of categorical attributes
    '''
    seperated=dataseperationByClasscategory(dataset)
    TrueFalseProbability={}
    for key,value in seperated.items():
        value=np.array(value)
        TrueFalseProbability[key]=thetaForEachClass(value,key)


    return TrueFalseProbability




def dataseperationByClasscategory(dataset):
    '''

    : Seperating the categoric attribute's tuples for each class Value
    :param dataset: the entire dataset
    :return: tuplesofeachclass : tuples for each class Value
    '''
    tuplesofeachclass = {}
    i=0
    while i<len(dataset):
        temp = dataset[i]
        if (temp[6] not in tuplesofeachclass):
            tuplesofeachclass[temp[6]] = []
        tuplesofeachclass[temp[6]].append(temp)
        i=i+1
    return tuplesofeachclass

def thetacalculationTrue(ValuesofAttributes,dataset):
    '''
    Calcualting the likelihood when class value is True
    :param ValuesofAttributes: values in the attribute
    :param dataset: The entire dataset
    :return: result2,result1 : the likelihood when class value is True
    '''
    countTrue=0
    countFalse=0
    totalcountofclass=len(dataset)
    dataset=np.array(dataset)

    for i in range(0,len(ValuesofAttributes)):
        if dataset[i][-1] =="True" and ValuesofAttributes[i] =="True":
            countTrue=countTrue+1
        if dataset[i][-1] == "True" and ValuesofAttributes[i] == "False":
            countFalse = countFalse + 1

    result1 = countTrue / totalcountofclass
    result2 = countFalse / totalcountofclass
    return result2,result1

def thetacalculationFalse(ValuesofAttributes,dataset):
    '''
    Calcualting the likelihood when class value is False
    :param ValuesofAttributes: values in the attribute
    :param dataset: The entire dataset
    :return: result2,result1 : the likelihood when class value is False
    '''
    countFalse=0
    countTrue=0
    totalcountofclass=len(dataset)
    dataset=np.array(dataset)

    for i in range(0,len(ValuesofAttributes)):
        if dataset[i][-1] =="False" and ValuesofAttributes[i] =="False":
            countFalse=countFalse+1
        if dataset[i][-1] == "False" and ValuesofAttributes[i] == "True":
            countTrue = countTrue + 1


    result1 =countFalse/totalcountofclass
    result2 =countTrue/totalcountofclass
    return result1,result2

def thetaForEachClass(dataset,key):
  '''

  :param dataset: The entire dataset
  :param key: Each class value
  :return: dict_of_theta_By_Class : dictionary having probabilities seperated by class
  '''
  dict_of_theta_By_Class={}
  for i in range (0,6):
    pos=i
    attribute=dataset[:, pos]  # 0 --> 6

    if key=="False":
     result1,result2=thetacalculationFalse(attribute,dataset)
    if key =="True":

     result1,result2=thetacalculationTrue(attribute,dataset)


    dict_of_theta_By_Class[pos]=(result1,result2)
  return dict_of_theta_By_Class


if __name__ == '__main__':
    main()
