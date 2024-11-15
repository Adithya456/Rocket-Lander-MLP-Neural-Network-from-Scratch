#Data preprocessing
import pandas as pd
#Reading CSV file
data_file=pd.read_csv("C:/Users/nagad/Desktop/NeuralNetworks_WorkSpace/ce889_dataCollection.csv",header=None)
#Checking info
data_file.info()
#Droping duplicate values
data_file.drop_duplicates()
#Checking for null values
data_file.isnull().sum()
#Droping "na" values
data_file.dropna()
#min values
data_file.min()
#max values
data_file.max()
#Performing Normalization on dataframe to scale the values between 0 and 1
data_file=data_file.apply(lambda x:(x-x.min())/(x.max()-x.min()))
#Creating a variable to use it for dividing dataset into train and test and validation datasets
#Assigning the variable with 70% value of length of dataset 
x=int((70/100)*len(data_file))
#Assigning the variable with 15% value of length of dataset 
y=int((15/100)*len(data_file))
#Shuffling the csv file to divide data file into  training and testing sets
shuffled_data_file=data_file.sample(frac = 1)
#Dividing dataset into training and testing datasets
training_dataset=shuffled_data_file[0:x]
validation_dataset=shuffled_data_file[x:x+y]
test_dataset=shuffled_data_file[x+y:]
#Converting the train and test_validation dataframe to csv files
training_dataset.to_csv("C:/Users/nagad/Desktop/NeuralNetworks_WorkSpace/Train_Dataset.csv", index=False, header=False)
validation_dataset.to_csv("C:/Users/nagad/Desktop/NeuralNetworks_WorkSpace/Validation_Dataset.csv", index=False, header=False)
test_dataset.to_csv("C:/Users/nagad/Desktop/NeuralNetworks_WorkSpace/Test_Dataset.csv", index=False, header=False)