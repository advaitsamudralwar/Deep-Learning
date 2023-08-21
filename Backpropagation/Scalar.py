import numpy as np
import argparse
import os

#Ref1:: https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd

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

def scalarNeuralnet(X,Y,W1,W2,W3,actfn):
  
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
    dldy = 2 * (actoutputlayer-Y) 
    # print("dldy", dldy)
    dldo3 = dldy * (diff1)
    dldw3 = np.dot((actout2), dldo3) 
    # print("grad_w3", dldw3)

    #--------second last layer backprop
    dldo2 = np.dot(dldo3, (W3))
    diff2 = activationDerivative(actfn, out2)
    dldh2 = dldo2 * (diff2) 
    # print("dldh2", dldh2)
    dldw2 = np.dot((actout1), dldh2)
    # print("dldw2",dldw2)

    dldo1 = np.dot(dldh2, (W2))
    diff3 = activationDerivative(actfn, out1)
    dldh1 = dldo1 * (diff3)
    dldw1 = (X) * dldh1
    # print("dldh1", dldh1)
    # print("dldw1",dldw1)
   
    return dldw3, dldw2, dldw1



if __name__ == '__main__':

    #-------------Get input and output file direc
    inputs = argparse.ArgumentParser()
    inputs.add_argument('--input', type=str)
    inputs.add_argument('--output', type=str)
    param = inputs.parse_args()
    neural_params = np.loadtxt(param.input, dtype=str)

    #-------------Set x,y , w1, w2, w3 valuclear
    # es
    x = float(neural_params[0])
    y = float(neural_params[1])
    w1 = float(neural_params[2])
    w2 = float(neural_params[3])
    w3 = float(neural_params[4])
    actfn = neural_params[5]

    print(x,y,w1,w2,w3)

    g3, g2, g1 = scalarNeuralnet(x,y,w1,w2,w3,actfn)
    
    # print(g3, g2, g1)
    np.savetxt(param.output, (g1,g2,g3))

    

    # python task_1.py --input "/Users/advaitsamudralwar/Desktop/UB Classes/Deep Learning/Assignment 2/input.txt" --output "/Users/advaitsamudralwar/Desktop/UB Classes/Deep Learning/Assignment 2/output.txt"
