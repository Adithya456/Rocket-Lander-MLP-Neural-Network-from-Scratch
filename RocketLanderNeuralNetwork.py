#Neuron Class
import math
import numpy as np
import random
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
class NeuralNetClass:
    def __init__(self):
        pass

    #Random Weights Generation Function
    def generateWeights(self,op_neurons,ip_neurons):
        weights=[]
        for i in range(ip_neurons):
            weights.append([0]*op_neurons)
        for j in range(ip_neurons):
            for k in range(op_neurons):
                weights[j][k]=random.random()
        return weights

    #Activation Function
    def activationFunction(self,v):
        x=1/(1+math.exp(-0.5*v))
        return x
    
    #Weight Multiplication Function
    def weightMultiplication(self,inpts,weights):
        v=np.dot(inpts,weights)
        return v

    #Feedforward Function
    def feedForward(self,values,bias_weights):
        outputs=[]
        for i in range(len(values)):
            h1=self.activationFunction(values[i]+bias_weights[0][i])
            outputs.append(h1)
        return outputs
        
    #Error Function
    def errorCalculation(self,actual_outputs,predicted_outputs):
        e=[]
        for i in range(len(predicted_outputs)):
            e1=actual_outputs[i]-predicted_outputs[i]
            e.append(e1)
        return e
        
    #Output Layer Local Gradient Function
    def outputLayerLocalGradient(self,lmbda,predicted_outputs,errors):
        output_lg=[]
        for i in range(len(errors)):
            x=lmbda*predicted_outputs[i]*(1-predicted_outputs[i])*errors[i]
            output_lg.append(x)
        return output_lg
    
    #Hidden Layer Local Gradient Function
    def hiddenLayerLocalGradient(self,lmbda,h,output_lg,outputWeights):
        #n=[]
        hidden_lg=[]
        outputWeights=np.array(outputWeights).transpose().tolist()
        n=np.dot(output_lg,outputWeights)
        for i in range(len(h)):
            m=lmbda*h[i]*(1-h[i])*n[i]
            hidden_lg.append(m)
        return hidden_lg
    
    #Delta Weights Function
    def DeltaWeights(self,eta,alfa,lg,inpts,prev_Delta_weights):
        deltaWeights=[]
        prev_Delta_weights=np.array(prev_Delta_weights).transpose().tolist()
        for i in range(len(lg)):
            n=[]
            for j in range(len(inpts)):
                Delta_W=(eta*lg[i]*inpts[j])+(alfa*prev_Delta_weights[i][j])
                n.append(Delta_W)
            deltaWeights.append(n)
        deltaWeights=np.array(deltaWeights).transpose().tolist()
        return deltaWeights

    #Weight Updation Function
    def weightUpdating(self,weights,deltaWeights):
        updatedWeights=[]
        for i in range(len(weights)):
            z=[]
            for j in range(len(weights[0])):
                x=weights[i][j]+deltaWeights[i][j]
                z.append(x)
            updatedWeights.append(z)
        return updatedWeights


if __name__ =='__main__':
        #File Reading Function       
    def readingCSVFile(fileName):
        list1=[]
        with open(fileName) as df:  
            df1 = csv.reader(df)
            next(df1)
            for i in df1:
                list1.append(i)
            return list1

    #RMSE Error Calculation Function  
    def RMSEValue(errors):
        sumOfSquareOfErrors1=0
        sumOfSquareOfErrors2=0
        for i in range(len(errors)):
            for j in range(len(errors[i])-1):                                      #Sum of Square of errors_1(SE)
                sumOfSquareOfErrors1+=(errors[i][j])**2                             
        for i in range(len(errors)):
            for j in range(1,len(errors[i])):                                      #Sum of Square of errors_2(SE)
                sumOfSquareOfErrors2+=(errors[i][j])**2
        meanOfSquareOfError1=sumOfSquareOfErrors1/len(errors)                      #Mean of Sum of Square of errors_1(SE)
        meanOfSquareOfError2=sumOfSquareOfErrors2/len(errors)                      #Mean of Sum of Square of errors_2(SE)
        rootMeanSquareError1=math.sqrt(meanOfSquareOfError1)                       #RMSE_1
        rootMeanSquareError2=math.sqrt(meanOfSquareOfError2)                       #RMSE_2
        rootMeanSquareError= (rootMeanSquareError1+rootMeanSquareError2)/2         #Root Mean Square Error(RMSE)
        return rootMeanSquareError

    #NEURAL NETWORK ARCHITECTURE                                   
    num_of_inputs=2
    num_of_ouputs=2
    num_of_hidden_neurons=6                 #int(input('Enter number of hidden layer NEURONS : '))
    num_of_epochs=150                       #int(input("Enter number of Epochs : "))
    train_rmse_values=[]                    #Variable for Training RMSE Values
    validation_rmse_values=[]               #Variable for Validation RMSE Values
    lmbda=0.8                               #Lambda Value
    eta=0.05                                #Learning Rate  
    alfa=0.9                                #Momentum Rate                             
    counter=0                               #Counter variable for Early Stopping

    #Calling Neuron Class
    nn=NeuralNetClass()
    #Generating Random Weights for input layer
    input_weights=nn.generateWeights(num_of_hidden_neurons,num_of_inputs)
    #Generating Random Bias Weights for input layer
    bias_input_weights=nn.generateWeights(num_of_hidden_neurons,1)
    #Generating Random Weights for hidden layer
    hidden_weights=nn.generateWeights(num_of_ouputs,num_of_hidden_neurons)
    #Generating Random Bias Weights for hidden layer
    bias_hidden_weights=nn.generateWeights(num_of_ouputs,1)
    #Creating list with zeros as values for input_delta_weights
    prev_input_delta_weights=[]
    for a in range(len(input_weights)):
            prev_input_delta_weights.append([0]*len(input_weights[0]))
    #Creating list with zeros as values for hidden_delta_weights
    prev_hidden_delta_weights=[]
    for a in range(len(hidden_weights)):
            prev_hidden_delta_weights.append([0]*len(hidden_weights[0]))
    #Creating list with zeros as values for bias_input_delta_weights
    prev_bias_input_delta_weights=[]
    for a in range(len(bias_input_weights)):
            prev_bias_input_delta_weights.append([0]*len(bias_input_weights[0]))
    #Creating list with zeros as values for bias_hidden_delta_weights
    prev_bias_hidden_delta_weights=[]
    for a in range(len(bias_hidden_weights)):
            prev_bias_hidden_delta_weights.append([0]*len(bias_hidden_weights[0]))


    #For loop for number of epochs
    for j in tqdm(range(num_of_epochs)):
        #TRAINING PART
        train_data=readingCSVFile('Train_Dataset.csv')    
        train_errors=[]                             #Variable for training error Values
        train_actalOutput_list=[]                   #Variable for training set actual outputs
        for i in range(len(train_data)):
            inputData=train_data[i]
            input1=float(inputData[0])
            input2=float(inputData[1])
            inputs=[input1,input2]
            output1=float(inputData[3])
            output2=float(inputData[2])
            actual_outputs=[output1,output2]        #Actual Outputs
            train_actalOutput_list.append([output1,output2])

            #Feedforward from input layer to hidden layer
            hidden_Unactiated_Values=nn.weightMultiplication(inputs,input_weights)
            hidden_Values=nn.feedForward(hidden_Unactiated_Values,bias_input_weights)
            #Feedforward from hidden layer to output layer
            output_Unactiated_Values=nn.weightMultiplication(hidden_Values,hidden_weights)
            predicted_Outputs=nn.feedForward(output_Unactiated_Values,bias_hidden_weights)
            #Error calculation
            errors=nn.errorCalculation(actual_outputs,predicted_Outputs)
            train_errors.append(errors)

            #Back Propagation
            #Output layer local gradient calculation
            output_lg=nn.outputLayerLocalGradient(lmbda,predicted_Outputs,errors)
            #Weight Updation -> Output layer to hidden layer
            hidden_delta_weights=nn.DeltaWeights(eta,alfa,output_lg,hidden_Values,prev_hidden_delta_weights)
            updated_hidden_weights=nn.weightUpdating(hidden_weights,hidden_delta_weights)
            bias_hidden_delta_weights=nn.DeltaWeights(eta,alfa,output_lg,[1],prev_bias_hidden_delta_weights)
            updated_bias_hidden_weights=nn.weightUpdating(bias_hidden_weights,bias_hidden_delta_weights)
            #Hidden layer local gradient calculation
            hidden_lg=nn.hiddenLayerLocalGradient(lmbda,hidden_Values,output_lg,hidden_weights)
            #Weight Updation -> hidden layer to input layer
            input_delta_weights=nn.DeltaWeights(eta,alfa,hidden_lg,inputs,prev_input_delta_weights)
            updated_input_weights=nn.weightUpdating(input_weights,input_delta_weights)
            bias_input_delta_weights=nn.DeltaWeights(eta,alfa,hidden_lg,[1],prev_bias_input_delta_weights)
            updated_bias_input_weights=nn.weightUpdating(bias_input_weights,bias_input_delta_weights)
            
            #Assigning trained weights to original weight varibles after each row
            input_weights=updated_input_weights
            hidden_weights=updated_hidden_weights
            bias_input_weights=updated_bias_input_weights
            bias_hidden_weights=updated_bias_hidden_weights

            #Updating prev_delta_weights variables with current delta_weights values after each row
            prev_input_delta_weights=input_delta_weights
            prev_hidden_delta_weights=hidden_delta_weights
            prev_bias_input_delta_weights=bias_input_delta_weights
            prev_bias_hidden_delta_weights=bias_hidden_delta_weights
        
        #Trained weights after each epoch
        input_weights=updated_input_weights
        hidden_weights=updated_hidden_weights
        bias_input_weights=updated_bias_input_weights
        bias_hidden_weights=updated_bias_hidden_weights
        #Updating prev_delta_weights after each epoch
        prev_input_delta_weights=input_delta_weights
        prev_hidden_delta_weights=hidden_delta_weights
        prev_bias_input_delta_weights=bias_input_delta_weights
        prev_bias_hidden_delta_weights=bias_hidden_delta_weights

        #VALIDATION PART
        validation_data=readingCSVFile('Validation_Dataset.csv')
        validation_errors=[]                            #Variable for training error Values
        validation_actualOutput_list=[]                 #Variable for validation set actual outputs
        for i in range(len(validation_data)):
            inputData=validation_data[i]
            input1=float(inputData[0])
            input2=float(inputData[1])
            inputs=[input1,input2]
            output1=float(inputData[3])
            output2=float(inputData[2])
            actual_outputs=[output1,output2]            #Actual Outputs
            validation_actualOutput_list.append([output1,output2])

            #Feedforward from input layer to hidden layer
            hidden_Unactiated_Values=nn.weightMultiplication(inputs,input_weights)
            hidden_Values=nn.feedForward(hidden_Unactiated_Values,bias_input_weights)
            #Feedforward from hidden layer to output layer
            output_Unactiated_Values=nn.weightMultiplication(hidden_Values,hidden_weights)
            predicted_Outputs=nn.feedForward(output_Unactiated_Values,bias_hidden_weights)
            #Error calculation
            errors=nn.errorCalculation(actual_outputs,predicted_Outputs)
            validation_errors.append(errors)


        #RMSE calculation for both Train and Validation sets
        train_rmse_OfEachEpoch=RMSEValue(train_errors)
        validation_rmse_OfEachEpoch=RMSEValue(validation_errors)

        train_rmse_values.append(train_rmse_OfEachEpoch)
        validation_rmse_values.append(validation_rmse_OfEachEpoch)

        #Early Stopping
        if round(validation_rmse_values[j],5)>=round(validation_rmse_values[j-1],5):
            counter+=1
        else:
            counter=0
        if counter==5:
            print("Stopped Early at Epochs = ",j)
            break

    #Final Trained Weights
    trained_input_weights=updated_input_weights
    trained_hidden_weights=updated_hidden_weights
    trained_bias_input_weights=updated_bias_input_weights
    trained_bias_hidden_weights=updated_bias_hidden_weights


    #Testing Part
    input_weights=updated_input_weights
    hidden_weights=updated_hidden_weights
    test_rmse_values=[]  
    test_data=readingCSVFile('Test_Dataset.csv')
    test_errors=[]                              #Variable for training error Values
    test_actualOutput_list=[]                   #Variable for validation set actual outputs
    for i in range(len(test_data)):
        inputData=test_data[i]
        input1=float(inputData[0])
        input2=float(inputData[1])
        inputs=[input1,input2]
        output1=float(inputData[3])
        output2=float(inputData[2])
        actual_outputs=[output1,output2]         #Actual Outputs
        test_actualOutput_list.append([output1,output2])

        #Feedforward from input layer to hidden layer
        hidden_Unactiated_Values=nn.weightMultiplication(inputs,input_weights)
        hidden_Values=nn.feedForward(hidden_Unactiated_Values,bias_input_weights)
        #Feedforward from hidden layer to output layer
        output_Unactiated_Values=nn.weightMultiplication(hidden_Values,hidden_weights)
        predicted_Outputs=nn.feedForward(output_Unactiated_Values,bias_hidden_weights)
        #Error calculation
        errors=nn.errorCalculation(actual_outputs,predicted_Outputs)
        test_errors.append(errors)

    #RMSE calculation for Test set
    test_rmse_OfEachEpoch=RMSEValue(test_errors)
    test_rmse_values.append(train_rmse_OfEachEpoch)
    print("Validation RMSE is : ",validation_rmse_values[-1])
    print("Test RMSE is : ",test_rmse_values)
   
    #Plotting RMSE Graph
    plt.plot(train_rmse_values)
    plt.plot(validation_rmse_values)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(["train_rmse", "val_rmse"], loc ="upper right")
    plt.show()

    #Writing trained weights to csv file
    def writingToFile():
        data = {
        'numberOfHiddenLayerNeurons': [num_of_hidden_neurons],
        'TrainedInputWeights': [trained_input_weights],
        'TrainedHiddenWeights': [trained_hidden_weights],
        'TrainedBiasInputWeights': [trained_bias_input_weights],
        'TrainedBiasHiddenWeights': [trained_bias_hidden_weights]
        }
        # Converting into data frame
        df = pd.DataFrame(data)
        df.to_csv('TrainedWeightsAndNumberOfNeurons.csv', index=False, header=False)

    writingToFile()