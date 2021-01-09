# traffic-sign-multiclassification
## Introduction  
Traffic sign detection and recognition have been gaining growing attention from the past few years. Traffic signs hold important information that drivers can disregard due to fatigue driving or threatening weather conditions. Nowadays in unsuitable traffic conditions, drivers can not see the traffic signs, which can result in a lot of accidents. In these situations, the need for automatic detection and recognition of traffic signs emerges which helps the driver get an idea of traffic on the road ahead. An automatic traffic sign recognition system can identify and acknowledge images recorded from vehicle cameras and sensors. The main objective of this project was to improve the robustness and reliability of the framework for the identification and recognition of traffic signs that can be implemented in real-time.  

## Dataset  
The data set taken for this project has been enriched with the German traffic sign information. Road signs have several distinguishing features that they are graded based on. The data images in this dataset have been grouped into 43 classes according to their shapes and colors. The dataset contains a total of over 50,000 images. The images are further split up into Training data (39209 images) and Testing data (12630 images). The below plot shows the distribution of Train class and Test class images, respectively. 
 
 
It was evident that training data had enough number of images required to train the model and the model can be verified against testing data. 

A file named 'Meta.csv' contains meta-information about the classes in this dataset. This file functions as a collection of ground rules needed to detect and classify the image.  

## Methods used to classify images  
Keras deep learning framework was used to implement a convolutional neural network for image classification.  The CNN is a multi-layer neural network, which extracts features by combining convolution, pooling, and activation layers.  

### First model:  
The first model had the following architecture:  
•	1 Conv2D layer (filter=32, kernel_size=filters, activation= “relu”)  
•	MaxPool2D layer ( pool_size=(2,2))  
•	Flatten layer  
•	Dense Fully connected layer (500 nodes, activation= “relu”)  
•	Dense layer (43 nodes, activation= “softmax”)  
The model was built using filters of size 3x3, 5x5, 9x9 and 13x13. The model was trained on a batch size of 5 with an equivalent epoch  

Finally, the above model was enhance to build a feed-forward network with 6 convolution layers followed by a full connected hidden layer, with dropout layers in between. The model architecture is as follows:  

### Main model:  
•	2 Conv2D layer (filter=32, kernel_size=(5,5), activation= “relu”)  
•	MaxPool2D layer (pool_size=(2,2))  
•	Dropout layer (rate=0.25)  
•	2 Conv2D layer (filter=64, kernel_size=(3,3), activation= “relu”)  
•	MaxPool2D layer ( pool_size=(2,2))  
•	Dropout layer (rate=0.25)  
•	Flatten layer to squeeze the layers into 1 dimension  
•	Dense Fully connected layer (256 nodes, activation= “relu”)  
•	Dropout layer (rate=0.5)  
•	Dense layer (43 nodes, activation= “softmax”)  

As it can be seen all the layers have ‘relu’ activation except the output layers. The output layer uses ‘softmax' function as it outputs the probability for each of the target classes. Both the models were implemented using Sequential, Keras container to form a linear stack of layers.  
The images in our dataset were scaled down to 30x30 pixels so that each input image has the same resolution. We expected to visualize how accuracy and loss will change with time. Also, in order to improve the accuracy, we added a Learning rate scheduler implemented using Keras callback feature.  

## Experimental Results  
On comparing the training accuracy and validation accuracy with different filters, it was found that filter or kernel of size 3x3 performed the best since our data was also of the equivalent shape. The main CNN model achieved a training accuracy of 99%. It was then tested to predict the images from the actual test data and when an image was given to predict from the test data set the results printed were accurate.
