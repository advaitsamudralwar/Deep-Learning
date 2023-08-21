# Deep-Learning

## Texture Classification on DTD dataset using CNN

### Introduction
This repository contains the implementation and analysis of Convolutional Neural Networks (CNNs) for classifying textures from the DTD dataset. The projectw aims to classify 47 unique texture classes defined in the dataset, with each class having 120 images.
The DTD dataset consists of 5640 images, split into 47 classes. Image sizes range between 300x300 and 640x640, and each image is resized to 300x300. The images contain at least 90 percent of the surface representing the category attribute.

## Backpropagation

### Scalar Input Backpropagation

In the context of a single scalar input, backpropagation involves the following steps:

Forward Pass: Compute the output using the input and weights.
Calculate Error: Calculate the error between the predicted output and the target output.
Backward Pass: Calculate gradients of the error with respect to the weights using the chain rule.

### Vector Input Backpropagation

When dealing with vector inputs, the process is similar, but matrix operations are used to handle multiple inputs simultaneously. This improves efficiency and is the basis for training deep neural networks.


