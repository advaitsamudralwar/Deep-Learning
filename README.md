# Deep-Learning

## Texture Classification on DTD dataset using CNN

### Introduction
This repository contains the implementation and analysis of Convolutional Neural Networks (CNNs) for classifying textures from the DTD dataset. The projectw aims to classify 47 unique texture classes defined in the dataset, with each class having 120 images.

### Dataset
The DTD dataset consists of 5640 images, split into 47 classes. Image sizes range between 300x300 and 640x640, and each image is resized to 300x300. The images contain at least 90 percent of the surface representing the category attribute.

### CNN Architecture
![Task 1: CNN Architecture](https://github.com/advaitsamudralwar/Deep-Learning/blob/main/Texture%20Classification(DTD)/output/my_optimizedCNN_model_architecure.png)

For the initial attempt, the model was trained with stochastic gradient descent using a batch size of 30, learning rate 0.001, and momentum 0.9.
- Train Loss: ![Train Loss for Task 1](https://github.com/advaitsamudralwar/Deep-Learning/blob/main/Texture%20Classification(DTD)/output/myoptimized_model_loss.png)
- Train Accuracy: ![Train Accuracy for Task 1](https://github.com/advaitsamudralwar/Deep-Learning/blob/main/Texture%20Classification(DTD)/output/myoptimized_model_trainacc.png)
- Testing Accuracy: ![Testing Accuracy for Task 1](https://github.com/advaitsamudralwar/Deep-Learning/blob/main/Texture%20Classification(DTD)/output/myoptimized_model_testingacc.png)

### Results and Observations

Deeper models hold the promise of improved performance, but they often come with the challenge of maintaining stability and achieving strong generalization. It's imperative to delve into critical considerations such as activation functions, strategic layer modifications, and transformations within the neural network.

The model's behavior is notably influenced by its intricate loss function, which exhibits heightened sensitivity. Furthermore, the model tends to be predisposed to initialization conditions. During the initial attempt, training commenced at approximately 63 percent, yet advancement was hindered, possibly due to encountering saddle points. Notably, akin behavior manifested when initiation commenced at 68 percent. These instances underscore the significance of initialization bias. Ultimately, the test accuracy achieved by the trained model was measured at 31.2 percent.


### Conclusion
The project provides insights into training CNNs for texture classification. Understanding initialization biases, generalization, and how managing model complexity is crucial for achieving desired performance.



