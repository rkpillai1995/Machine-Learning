__author__ = 'Rajkumar Pillai'
import numpy as np
import matplotlib.pyplot as plt

"""

CSCI-739:  Topics in Intelligent Systems 
Author: Rajkumar Lenin Pillai

Description:This program shows the different plots as mentioned in question-1
"""

for i in range(0,2):
    x, y, species = [], [], []
    feature_hist_hylaminuta_MFCC_10 = []
    feature_hist_hylaminuta_MFCC_17 = []
    feature_hist_HypsiboasCinerascens_MFCC_10 = []
    feature_hist_HypsiboasCinerascens_MFCC_17 = []

    ###Reading of file
    if i==0:
      print("Dataset: Frogs.csv")
      file_name = 'Frogs.csv'
    else:
      print("Dataset: Frogs-subsample.csv")
      file_name ='Frogs-subsample.csv'
    with open(file_name) as f:
      v = np.loadtxt(f, delimiter=",", dtype='str', comments="#", skiprows=1, usecols=(0,1,2))



    #### Strings necessary for labeling the output
    feauture_name_MFCC_10="MFCC_10_values"
    feauture_name_MFCC_17="MFCC_17_values"
    feauture_name_hylaminuta_MFCC_10="Hylaminuta_MFCC_10"
    feauture_name_hylaminuta_MFCC_17="Hylaminuta_MFCC_17"
    feature_name_HypsiboasCinerascens_MFCC_10="HypsiboasCinerascens_MFCC_10"
    feature_name_HypsiboasCinerascens_MFCC_17="HypsiboasCinerascens_MFCC_17"


    ### Storing the values of each feature in an array
    for i in range(0,len(v)):
      x.append(float(v[i][0]))
      y.append(float(v[i][1]))
      species.append(str(v[i][2]))



    for i in range(0,len(v)):
      if(v[i][2]=='HylaMinuta'):
        feature_hist_hylaminuta_MFCC_10.append(np.float64(v[i][0]))


    for i in range(0,len(v)):
      if(v[i][2]=='HylaMinuta'):
        feature_hist_hylaminuta_MFCC_17.append(np.float64(v[i][1]))

    for i in range(0, len(v)):
        if (v[i][2] == 'HypsiboasCinerascens'):
          feature_hist_HypsiboasCinerascens_MFCC_10.append(np.float64(v[i][0]))

    for i in range(0, len(v)):
        if (v[i][2] == 'HypsiboasCinerascens'):
          feature_hist_HypsiboasCinerascens_MFCC_17.append(np.float64(v[i][1]))



    ############### Scatterplot #######################################################
    print("Scatter Plot")
    fig, ax = plt.subplots()

    ax.scatter(feature_hist_hylaminuta_MFCC_10, feature_hist_hylaminuta_MFCC_17, marker="o", color='orange', label="Hylaminuta")
    plt.legend()

    ax.scatter(feature_hist_HypsiboasCinerascens_MFCC_10, feature_hist_HypsiboasCinerascens_MFCC_17, marker="x", color='blue', label="HypsiboasCinerascens")
    plt.legend()
    plt.xlabel(feauture_name_MFCC_10)
    plt.ylabel(feauture_name_MFCC_17)
    plt.title('Scatterplot')
    plt.show()
    ###################################################################################



    ################################# Histogram ##########################################
    print("Histogram")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    ax1.grid(True)

    n, bins, patches = ax1.hist(feature_hist_hylaminuta_MFCC_10, bins=70, facecolor='blue',label="hist_hylaminuta_MFCC_10")
    #plt.legend()
    plt.xlabel(feauture_name_hylaminuta_MFCC_10)
    plt.ylabel("Frequency")
    plt.title('Histogram')
    plt.show()
    #################################################
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.grid(True)


    n, bins, patches = ax2.hist(feature_hist_hylaminuta_MFCC_17, bins=70, facecolor='blue')
    plt.xlabel(feauture_name_hylaminuta_MFCC_17)
    plt.ylabel("Frequency")
    plt.title('Histogram')
    plt.show()



    ############## Histogram MFCCCCCCC17 ################################################
    fig = plt.figure()
    ax3 = fig.add_subplot(111)

    ax3.grid(True)

    n, bins, patches = ax3.hist(feature_hist_HypsiboasCinerascens_MFCC_10, bins=70, facecolor='blue')
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_10)
    plt.ylabel("Frequency")
    plt.title('Histogram')
    plt.show()

    ###############################
    fig = plt.figure()
    ax4 = fig.add_subplot(111)
    ax4.grid(True)


    n, bins, patches = ax4.hist(feature_hist_HypsiboasCinerascens_MFCC_17, bins=70, facecolor='blue')
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_17)
    plt.ylabel("Frequency")
    plt.title('Histogram')
    plt.show()
    ##########################################################################################



    ################################ Line Graph ##############################################
    print("Line graph")


    feature_hist_hylaminuta_MFCC_10=np.sort(feature_hist_hylaminuta_MFCC_10)


    plt.plot(feature_hist_hylaminuta_MFCC_10)
    plt.xlabel(feauture_name_hylaminuta_MFCC_10)
    plt.ylabel(feauture_name_MFCC_10)
    plt.title('Line graph')
    plt.show()
    ################################

    feature_hist_hylaminuta_MFCC_17=np.sort(feature_hist_hylaminuta_MFCC_17)
    plt.plot(feature_hist_hylaminuta_MFCC_17)
    plt.xlabel(feauture_name_hylaminuta_MFCC_17)
    plt.ylabel(feauture_name_MFCC_17)
    plt.title('Line Graph')
    plt.show()


    #############################


    feature_hist_HypsiboasCinerascens_MFCC_10=np.sort(feature_hist_HypsiboasCinerascens_MFCC_10)
    plt.plot(feature_hist_HypsiboasCinerascens_MFCC_10)
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_10)
    plt.ylabel(feauture_name_MFCC_10)
    plt.title('Line Graph')
    plt.show()


    ############################


    feature_hist_HypsiboasCinerascens_MFCC_17=np.sort(feature_hist_HypsiboasCinerascens_MFCC_17)
    plt.plot(feature_hist_HypsiboasCinerascens_MFCC_17)
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_17)
    plt.ylabel(feauture_name_MFCC_17)
    plt.title('Line Graph')
    plt.show()
    #########################################################################################################

    ############################################## Box Plot #################################################
    print("Box Plot")


    fig = plt.figure()
    ax5 = fig.add_subplot(111)
    bp = ax5.boxplot(feature_hist_hylaminuta_MFCC_10)
    plt.xlabel(feauture_name_hylaminuta_MFCC_10)
    plt.ylabel(feauture_name_MFCC_10)
    plt.title('Box Plot')
    plt.show()
    ##############################################
    fig = plt.figure()
    ax5 = fig.add_subplot(111)
    bp = ax5.boxplot(feature_hist_hylaminuta_MFCC_17)
    plt.xlabel(feauture_name_hylaminuta_MFCC_17)
    plt.ylabel(feauture_name_MFCC_17)
    plt.title('Box Plot')
    plt.show()
    ##############################################
    fig = plt.figure()
    ax5 = fig.add_subplot(111)
    bp = ax5.boxplot(feature_hist_HypsiboasCinerascens_MFCC_10)
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_10)
    plt.ylabel(feauture_name_MFCC_10)
    plt.title('Box Plot')
    plt.show()
    ##############################################
    fig = plt.figure()
    ax5 = fig.add_subplot(111)
    bp = ax5.boxplot(feature_hist_HypsiboasCinerascens_MFCC_17)
    plt.xlabel(feature_name_HypsiboasCinerascens_MFCC_17)
    plt.ylabel(feauture_name_MFCC_17)
    plt.title('Box Plot')
    plt.show()
    #################################################################################################




    ################################ Bar Plot#################################################


    Hylaminuta_MFCC_10_mean = np.mean(feature_hist_hylaminuta_MFCC_10)
    Hylaminuta_MFCC_17_mean = np.mean(feature_hist_hylaminuta_MFCC_17)

    HypsiboasCinerascens_MFCC_10_mean = np.mean(feature_hist_HypsiboasCinerascens_MFCC_10)
    HypsiboasCinerascens_MFCC_17_mean = np.mean(feature_hist_HypsiboasCinerascens_MFCC_17)


    Hylaminuta_MFCC_10_std = np.std(feature_hist_hylaminuta_MFCC_10)
    Hylaminuta_MFCC_17_std = np.std(feature_hist_hylaminuta_MFCC_17)

    HypsiboasCinerascens_MFCC_10_std = np.std(feature_hist_HypsiboasCinerascens_MFCC_10)
    HypsiboasCinerascens_MFCC_17_std = np.std(feature_hist_HypsiboasCinerascens_MFCC_17)



    materials = ['Hylaminuta_MFCC_10']
    x_pos = np.arange(len(materials))
    y_pos = [Hylaminuta_MFCC_10_mean]
    error = [Hylaminuta_MFCC_10_std]

    fig, ax6 = plt.subplots()
    ax6.bar(x_pos, y_pos, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(materials)
    ax6.yaxis.grid(True)
    plt.tight_layout()
    plt.title("Bar Plot")
    plt.ylabel(feauture_name_MFCC_10)
    plt.show()
    #######################################


    materials = ['Hylaminuta_MFCC_17']
    x_pos = np.arange(len(materials))
    y_pos = [Hylaminuta_MFCC_17_mean]
    error = [Hylaminuta_MFCC_17_std]

    fig, ax6 = plt.subplots()
    ax6.bar(x_pos, y_pos, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(materials)
    ax6.yaxis.grid(True)
    plt.tight_layout()
    plt.title("Bar Plot")
    plt.ylabel(feauture_name_MFCC_17)
    plt.show()
    ######################################

    materials = ['HypsiboasCinerascens_MFCC_10']
    x_pos = np.arange(len(materials))
    y_pos = [HypsiboasCinerascens_MFCC_10_mean]
    error = [HypsiboasCinerascens_MFCC_10_std]

    fig, ax6 = plt.subplots()
    ax6.bar(x_pos, y_pos, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(materials)
    ax6.yaxis.grid(True)
    plt.tight_layout()
    plt.title("Bar Plot")
    plt.ylabel(feauture_name_MFCC_10)
    plt.show()

    ###############################################

    materials = ['HypsiboasCinerascens_MFCC_17']
    x_pos = np.arange(len(materials))
    y_pos = [HypsiboasCinerascens_MFCC_17_mean]
    error = [HypsiboasCinerascens_MFCC_17_std]

    fig, ax6 = plt.subplots()
    ax6.bar(x_pos, y_pos, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(materials)
    ax6.yaxis.grid(True)
    plt.tight_layout()
    plt.title("Bar Plot")
    plt.ylabel(feauture_name_MFCC_17)
    plt.show()





    #########################################################################################################################

    ######################## Covariance #############################

    cov_matrix_x=np.array(x)
    #print("Cov",np.cov(cov_matrix_x))


    cov_matrix_y=np.array(y)
    #print("Cov",np.cov(cov_matrix_y))

    print("Covariance matrix")
    print(np.cov(cov_matrix_x,cov_matrix_y))

    ################################################################


    ########################## Mean and standard deviation######################################
    MFCC_mean_10 = np.mean(x)
    MFCC_std_10 = np.std(x)

    MFCC_mean_17 = np.mean(y)
    MFCC_std_17 = np.std(y)

    print("Mean of MFCC_10 is: ",MFCC_mean_10)
    print("Standard Deviation of MFCC_10 is: ",MFCC_std_10)


    print("Mean of MFCC_17 is: ",MFCC_mean_17)
    print("Standard Deviation of MFCC_17 is: ",MFCC_std_17)







