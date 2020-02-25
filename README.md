# SELF-DRIVING CAR: Traffic Sign Classifier
One of the essential techniques in Self-Driving Car engineering is detecting the Traffic Sign. In [previous projects](https://github.com/PooyaAlamirpour/TrafficSignClassifier), we have talked about other critical parts. For acquiring to being a specialist in Self-Driving Car, we should know about recognizing the traffic sign and other obstacles on the road. So let us implement it in the following steps:
* Visualizing the database
* Preprocessing the data within the dataset
* Design, train and validating a Deep Neural Network Architecture
* Using newfound images for testing the performance of our model

I have prepared a [video](https://www.youtube.com/watch?v=fJYYmyuJm9I&list=PLChwywmfd8lqhyap8yrjOeALFLkJ5nRTv&index=2&t=108s) from my implementation that you can watch it too.

### Visualizing the database
There is a folder that is called `dataset,` and it is uploaded the used dataset for this project. So it can be downloadable for use on a personal project. However, before that, for more precise, let us walk-through on images that exist in this dataset.
This dataset has 43 categories for the traffic sign and has 809 images on average in each category. The minimum number of the image belongs to the Speed Limit (20Km/h) category that has 180 images, and the maximum number belongs to the Speed Limit (30Km/h) that has 2010 images. Additional detail can be found below:
* Number of the training images is 34799
* Number of the validation images is 4410
* Number of the test images is 12630

The size of all images is 32x32. Here is a plot that shows how the data is distributed based on each category.

<img src="https://github.com/PooyaAlamirpour/TrafficSignClassifier/blob/master/plots/number_of_samples_on_each_category.png" width="640" border="10" />

Here demonstrate some images that were randomly picked up from the dataset and were taken in different weather and light situations.

<img src="https://github.com/PooyaAlamirpour/TrafficSignClassifier/blob/master/plots/random_examples.png" width="640" border="10" />

### Preprocessing the data within the dataset
In this project, all images are converted to the YUV space, and just the Y channel is used. Because preliminary experience shows, using all channels would be caused to confuse the traffic sign recognizer. Images are randomly perturbed in rotation and position. It is required to train the network to immune against this turbulence. For achieving that goal, some arbitrary images should be randomly rotated, transformed, and scaled. So the network can be trained in other styles of traffic sign images. The `preprocessImages` method does all these jobs. All this scenario have done for the training and validation sets.

### Design, train and validating a Deep Neural Network Architecture
In this project, a model is designed with ten layers that are connected by using the `Relu.`
The input layer has a 32x32x1 dimension and receives images in the Y channel. Then a 5x5 Convolution layer is connected to it by 1x1 stride and the valid padding. The size of the output of this layer is 28x28x6. After that, there is a max-pooling layer that is connected by using a Relu to the Convolution layer. The output of max-pooling with 2x2x stride is 16x16x64. Again another 5x5 Convolution with 1x1 stride and valid padding is added, so the size of the output of this layer is 14x14x6, then a Relu is added to the output of it. In the following, It is connected to a Fully-Connected, Relu, and Dropout layer. So 14x14x6 is converted to 120. Again it is used another Fully-Connected layer with Relu and Dropout. Now the size of the output of this model in this step is 84. Finally, a Fully-Connected layer with Sfotmax is connected to the last layer. So the output size of this model is 43.
This model is trained by using an Adam Optimizer. The learning rate of this model is 1e-4, and the rate of the dropout in all layers is 0.3. The size of the batch that is used in designing this model is 128.

<img src="https://github.com/PooyaAlamirpour/TrafficSignClassifier/blob/master/plots/learning_curve.png" width="640" border="10" />

One of the best architecture for detecting more than ten classes is LeNet. So starting from this point is a good idea. After a couple runs, this model demonstrated to tend to overfit on the train data. The question is, How is it possible to detect overfitting? Considering on Learning Curve can be perceived lots of important detail about the performance of a model. If the training curve shows to converge to the 99% accuracy and the validation curve does not have satisfactory performance, the model can be overfitting. Two techniques can survive a model from this circumstance.
* Adding more data
* Dropout

As it is mentioned, a method is implemented for randomly rotating, scaling, and transforming. It can generate more data, but the most effective way to surviving a model from overfitting is to use the Dropout procedure. It is observable that using a high dropout rate, around 50% causes the learning rate is decreased. In this project, it is chosen a 30% dropout based on try and error. During training the model, the training curve shows adding more data, and using the dropout method caused a strong increased performance while the number of Epochs can be decreased. 
Finally, the model has the accuracy of nearby 97% on the training and 95% on the validation and 93% on the test set.

### Using newfound images for testing the performance of our model
For evaluating a model, it is better to test it by using some new images outside of the dataset. Here are five German Traffic signs that can be used for testing.

<img src="https://github.com/PooyaAlamirpour/TrafficSignClassifier/blob/master/plots/newfound.png" width="640" border="10" />

These images have messy and different backgrounds or have turbulence surfaces by a watermark. So these would be great for appraising our model. At first, we should pass all images into `preprocessImages` function. Here are the top five softmax result for each image:
As the result shows, our model can detect four traffic signs correctly, and the accuracy of it is nearly 80%. It is entirely comprehensible why the model has lower than precision on newfound images. Because it does not have experience for learning newfound images.

```python
Top 5 Labels for image 'Children crossing':
 - 'Children crossing' with prob = 0.80 
 - 'Go straight or right' with prob = 0.08 
 - 'End of no passing' with prob = 0.06 
 - 'Keep right' with prob = 0.04 
 - 'No passing' with prob = 0.01 
Top 5 Labels for image 'Wild animals crossing':
 - 'Children crossing' with prob = 1.00 
 - 'Keep right' with prob = 0.00 
 - 'Dangerous curve to the right' with prob = 0.00 
 - 'Go straight or right' with prob = 0.00 
 - 'Slippery road' with prob = 0.00 
Top 5 Labels for image 'Stop':
 - 'Stop' with prob = 1.00 
 - 'Priority road' with prob = 0.00 
 - 'Yield' with prob = 0.00 
 - 'Turn right ahead' with prob = 0.00 
 - 'No entry' with prob = 0.00 
Top 5 Labels for image 'No vehicles':
 - 'No vehicles' with prob = 1.00 
 - 'Speed limit (30km/h)' with prob = 0.00 
 - 'Speed limit (50km/h)' with prob = 0.00 
 - 'Speed limit (80km/h)' with prob = 0.00 
 - 'No passing' with prob = 0.00 
Top 5 Labels for image 'Bumpy road':
 - 'Bumpy road' with prob = 1.00 
 - 'Bicycles crossing' with prob = 0.00 
 - 'Road work' with prob = 0.00 
 - 'Children crossing' with prob = 0.00 
 - 'Dangerous curve to the right' with prob = 0.00 
```

Thanks for reading my article and watching my implementation video. Feel free to ask me any questions.

### Refrence
* [HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
* [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)