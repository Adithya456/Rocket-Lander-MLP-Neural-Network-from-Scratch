from RocketLanderNeuralNetwork import NeuralNetClass
import pandas as pd
import ast
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        Trained_Data=pd.read_csv('C:/Users/nagad/Desktop/NeuralNetworks_WorkSpace/TrainedWeightsAndNumberOfNeurons.csv')
        Data=[]
        for i in Trained_Data:
            Data.append(ast.literal_eval(i))
        self.x1min= -730.141792   
        self.x1max= 510.405610        
        self.x2min= 65.511933        
        self.x2max= 644.658951    
        self.y1min= -6.679174     
        self.y1max=  7.826612     
        self.y2min= -4.346960      
        self.y2max=  7.981569  
        self.numberOfNeurons=Data[0]
        self.UpdatedInputWeights = Data[1]
        self.UpdatedHiddenWeights = Data[2]
        self.UpdatedBiasInputWeights=Data[3]
        self.UpdatedBiasHiddenWeights=Data[4]
    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        inputs=ast.literal_eval(input_row)
        #Normalization
        input1= (inputs[0]-self.x1min)/(self.x1max-self.x1min)
        input2= (inputs[1]-self.x2min)/(self.x2max-self.x2min)
        inputs=[input1,input2]
        #Feedforward
        nn=NeuralNetClass()
        input_weights=self.UpdatedInputWeights
        bias_input_weights=self.UpdatedBiasInputWeights
        hidden_Unactiated_Values=nn.weightMultiplication(inputs,input_weights)
        hidden_Values=nn.feedForward(hidden_Unactiated_Values,bias_input_weights)
        hidden_Weights=self.UpdatedHiddenWeights
        bias_hidden_weights=self.UpdatedBiasHiddenWeights
        output_Unactiated_Values=nn.weightMultiplication(hidden_Values,hidden_Weights)
        predicted_Outputs=nn.feedForward(output_Unactiated_Values,bias_hidden_weights)
        #Denormalization
        predicted_Outputs[0]=predicted_Outputs[0]*(self.y1max-self.y1min)+self.y1min
        predicted_Outputs[1]=predicted_Outputs[1]*(self.y2max-self.y2min)+self.y2min
        y=[predicted_Outputs[0],predicted_Outputs[1]]
        return y
    
        
