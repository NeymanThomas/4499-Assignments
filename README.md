# 4499-Assignments
4499 - Applied Neural Networks Homework and Project assignments are posted in this repository.

## Tabel of Contents
- [Homework 1](#homework-1)
- [Homework 2](#homework-2)
- [Homework 3](#homework-3)
- [Mid Project](#mid-project)

## [Homework 1](https://github.com/NeymanThomas/4499-Assignments/tree/main/Homework-1)
Import in the MNIST digits dataset (70,000 images, each 28x28 pixels). Preprocess the data as necessary. The training set should have 60k images and the test set 10k. 
Also create a validation set within the training set -- using 10% of the training data. </br>
### **Part 1**
Using the Keras Sequential API, create a neural network with two hidden layers of 500 neurons each. Train it for 30 epochs. Graph the accuracy and loss for the training and validation sets. Does your model become overfit? If so, at about what epoch? Re-train your neural network for that number of epochs. Now test your neural network on your testing data. What accuracy do you achieve? Is it about the same as the accuracy on the validation data? Plot several of the misclassified images from your model.
### **Part 2**
Try to improve your model by changing your architecture (try different numbers of neurons per layer, and/or more layers) and report again on accuracies and plot several misclassified images.
### **Part 3**
Which model did best? Discuss.

## [Homework 2](https://github.com/NeymanThomas/4499-Assignments/tree/main/Homework-2)
Import in the MNIST digits dataset (70,000 images, each 28x28 pixels). Preprocess the data as necessary. The training set should have 60k images and the test set 10k. Also create a validation set within the training set -- using 10% of the training data.
### **Part 1**
Create a deep and wide neural network using the Keras Functional API (exact architecture left up to you). Train it for an appropriate number of epochs with an appropriate learning rate and plot the accuracy/loss vs epoch. Find the accuracy on the test data and plot the first five misclassified images.
### **Part 2**
Use the subclassing API to create a different multi-path network than Part 1 (ie, a deep and wide network is multi-path) with two different inputs. Send all pixels in one input and a subset of pixels in the second input. Experiment with different architectures. Train it for an appropriate number of epochs with an appropriate learning rate and plot the accuracy/loss vs epoch. Find the accuracy on the test data and plot the first five misclassified images.
### **Part 3**
Which NN was the best? What was its accuracy? It's confusion matrix? Discuss.

## [Homework 3](https://github.com/NeymanThomas/4499-Assignments/tree/main/Homework-3)
Kaggle contains many useful datasets and data science competitions. It also has great tutorials and discussion boards. The data for this assignment comes from the [Kaggle Cats vs Dogs competition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview). Images such as these were once used for CAPTCHA (after digits and the alphabet proved too crackable). As stated in the overview, many years ago computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art (you should do better than this even without using CNNs). However, even back in 2014 state of the art machine learning could exceed 80% accuracy on this cat and dog dataset. This meant it was no longer useful for CAPTCHA. Currently, with the utilization of transfer learning this accuracy can exceed 95%.
### **Part 1**
Download the dataset. Go to [this link](https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/) and set up your computer and Colab to easily download Kaggle datasets. Then download the `dogs-vs-cats-redux-kernels-edition` dataset.
### **Part 2**
Preprocess the dataset (load it into one dataframe and create your y labels).
Print 5 sample images of dogs and cats (BEFORE they have been resized).
### **Part 3**
Now split the training data into training (15000 images), validation (5000 images), and testing (5000 images) datasets.
### **Part 4**
Try different NN architectures and options. Use KerasTuner (or alternatively the sklearn tools RandomizedSearchCV or GridSearchCV) as part of this. Try at least one deep neural network with at least 50 hidden layers. Clearly state initialization, activation, architecture (including # layers and neurons, and pathways), any normalization/regularization used, and other relevant information for each model.

You are NOT expected to utilize transfer learning, data augmentation, or convolutional neural networks (these will be added in a future assignment).

Give converged validation and testing accuracy for each model trained (utilizing the EarlyStopping callback). Which one was the best? Display five misclassified images from your best model. Why do you think it was the best? Note: Your grade for this homework will depend on the quality of your best model.

## [Mid Project](https://github.com/NeymanThomas/4499-Assignments/tree/main/Mid-Project)
Use the Cats and Dogs dataset in Kaggle from Homework 3. Download the dataset and perform necessary data wrangling or preprocessing.
### **Part 1**
Create a CNN dog/cat classifier with the following architecture: </br>
Conv2D: 64 filters, 7x7 kernel </br>
MaxPooling2D: 2x2 pool</br>
Conv2D: 128 filters, 3x3 kernel</br>
Conv2D: 128 filters, 3x3 kernel</br>
MaxPooling2D: 2x2 pool</br>
Conv2D: 256 filters, 3x3 kernel</br>
Conv2D: 256 filters, 3x3 kernel</br>
MaxPooling2D: 2x2 pool</br>
Flatten</br>
Dense: 128 neurons</br>
Dropout(15%)</br>
Dense: 64 neurons</br>
Dropout(15%)</br>
Output</br>
strides=1, padding="same", and activation="relu" for all relevant layers</br>
Utilize the Nadam optimizer and Early Stopping

What is your accuracy for the validation set and for the test set? Plot several misclassified images.

Save your accuracy and val_accuracy each epoch. You will plot them in Part 4.

### **Part 2**
Add data augmentation. Try different CNN architectures and/or parameters to improve your model. You may also try splitting up the dataset differently (ie more images in training and less in validation and testing sets).

What is the accuracy of your best model (on both validation and testing sets)? Plot several misclassified images. Save your accuracy and val_accuracy per epoch (from your best model). You will plot them in Part 4.

### **Part 3**
Use transfer learning. Train on at least three different expert CNNs: Xception, a variant of ResNet, and a variant of EfficientNet. See [here](https://keras.io/api/applications/) for a full list of models available in Keras. Please note that many of these have special requirements (for image size etc). You decide (or experiment with) how to best transfer learn on these expert CNNs.

What is the accuracy of your best model (on both validation and testing sets)? Plot several misclassified images. Save your accuracy and val_accuracy per epoch (from your best model). You will plot them in Part 4.

### **Part 4**
Plot the training and validation accuracy per epoch for your (best) model from each part. Discuss any differences. Which model(s) tend to be overfit? Which model(s) are not? Why?

Note: If you have a model B that is only a little better than a model A with respect to validation/testing accuracy, but model B is not overfit and model A is, that is actually a huge improvement.
