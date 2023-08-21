import numpy as np
import argparse
import os
import ast


#Ref 1: https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd

def activationFunction(actfn, input):
    if actfn == "Logistic":
        temp = 1 / (1 + np.exp(-input))
    elif actfn == "Relu":
        temp =np.maximum(0,input)
    else:
        temp = np.tanh(input)
    return temp

def activationDerivative(actfn, input):
    if actfn == "Logistic":
        temp = activationFunction(actfn, input)
        temp = temp * (1-temp)
    elif actfn == "Relu":
        temp = activationFunction(actfn, input)
        temp = np.where(temp > 0, 1, 0)
    else:
        temp = activationFunction(actfn, input)
        temp = 1 - np.square(np.tanh(input))
    return temp

def vectorNeuralnet(X,Y,W1,W2,W3,actfn):
  
    #-----------------------------------------------Forward-Prop
    #-------Output of fist layer after activation function(x as input)
    out1 = np.dot(X, W1)
    actout1 = activationFunction(actfn, out1)

    #-------Output of second layet after actication function(output of first layer will be the input)
    out2  = np.dot(actout1, W2)
    actout2 = activationFunction(actfn, out2)

    #-------Output of output layet after actication function(output of second layer will be the input)
    outputlayer  = np.dot(actout2, W3)
    actoutputlayer = activationFunction(actfn, outputlayer)

    #--------Mean squared loss
    error1 = np.mean(np.square(np.subtract(actoutputlayer,Y)))
    print("Predicted Y is:", actoutputlayer, "MSE: ", error1)
    

    #-----------------------------------------------Back-Prop

    #--------output layer backprop
    diff1 = activationDerivative(actfn, outputlayer)
    # print( diff1)
    dldy = 2 * (actoutputlayer-Y) / Y.size
    # print("dldy", dldy)
    dldo3 = dldy * (diff1)
    dldw3 = np.dot(np.transpose(actout2), dldo3) 
    # print("grad_w3", dldw3)

    #--------second last layer backprop
    dldo2 = np.dot(dldo3, np.transpose(W3))
    diff2 = activationDerivative(actfn, out2)
    dldh2 = dldo2 * (diff2) 
    # print("dldh2", dldh2)
    dldw2 = np.dot(np.transpose(actout1), dldh2)
    # print("dldw2",dldw2)

    dldo1 = np.dot(dldh2, np.transpose(W2))
    diff3 = activationDerivative(actfn, out1)
    dldh1 = dldo1 * (diff3)
    dldw1 = (np.transpose(X) * dldh1)
    # print("dldh1", dldh1)
    # print("dldw1",dldw1)
   
    return dldw3, dldw2, dldw1


if __name__ == '__main__':

    #-------------Get input and output file direc
    inputs = argparse.ArgumentParser()
    inputs.add_argument('--input', type=str)
    inputs.add_argument('--output', type=str)
    param = inputs.parse_args()
    neural_params = []

    #-----------Read data as arrays from txt files consisting of vectors
    #-----------Set values of X,Y, W1,W2,W3, and actfn
    with open(param.input) as f:
        data = f.readlines()
        neural_params = ([i.strip() for i in data])
    X = np.array(ast.literal_eval(neural_params[0]),dtype=float)
    Y = np.array(ast.literal_eval(neural_params[1]),dtype=float)
    W1 = np.array(ast.literal_eval(neural_params[2]),dtype=float)
    W2 = np.array(ast.literal_eval(neural_params[3]),dtype=float)
    W3 = np.array(ast.literal_eval(neural_params[4]),dtype=float)
    actfn = (neural_params[5])

    # print("Shapes are:",X.shape, Y.shape, W1.shape, W2.shape, W3.shape)
    # print("X, Y, W1, W2, W3, act",  X,  Y, W1, W2, W3, actfn)
    # change shape for forward and backprop
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)

    G3, G2, G1 = vectorNeuralnet(X,Y,W1,W2,W3,actfn)
    # print(G3,"\n", G2,"\n", G1)
    with open(param.output, 'w') as f:
            f.writelines([str(val) + '\n' for val in [G1, G2, G3]])
    
    # python task_1.py --input "/Users/advaitsamudralwar/Desktop/UB Classes/Deep Learning/Assignment 2/input1.txt" 



