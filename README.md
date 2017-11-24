# Traffic Sign Classifier

## Overview

The work for this project was done in a Jupyter notebook which
[can be reviewed here](https://github.com/laurentgauthier/p-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb).

The goals / steps of this project were the following:

* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[sign01]: ./personal-photos/traffic-sign-01.png "Traffic Sign 01"
[sign02]: ./personal-photos/traffic-sign-02.png "Traffic Sign 02"
[sign03]: ./personal-photos/traffic-sign-03.png "Traffic Sign 03"
[sign04]: ./personal-photos/traffic-sign-04.png "Traffic Sign 04"
[sign05]: ./personal-photos/traffic-sign-05.png "Traffic Sign 05"
[sign06]: ./personal-photos/traffic-sign-06.png "Traffic Sign 06"
[sign07]: ./personal-photos/traffic-sign-07.png "Traffic Sign 07"
[sign08]: ./personal-photos/traffic-sign-08.png "Traffic Sign 08"
[sign09]: ./personal-photos/traffic-sign-09.png "Traffic Sign 09"
[sign10]: ./personal-photos/traffic-sign-10.png "Traffic Sign 10"
[sign11]: ./personal-photos/traffic-sign-11.png "Traffic Sign 11"
[sign12]: ./personal-photos/traffic-sign-12.png "Traffic Sign 12"
[sign13]: ./personal-photos/traffic-sign-13.png "Traffic Sign 13"
[sign14]: ./personal-photos/traffic-sign-14.png "Traffic Sign 14"
[sign15]: ./personal-photos/traffic-sign-15.png "Traffic Sign 15"
[sign16]: ./personal-photos/traffic-sign-16.png "Traffic Sign 16"
[sign17]: ./personal-photos/traffic-sign-17.png "Traffic Sign 17"
[sign18]: ./personal-photos/traffic-sign-18.png "Traffic Sign 18"
[sign19]: ./personal-photos/traffic-sign-19.png "Traffic Sign 19"
[sign20]: ./personal-photos/traffic-sign-20.png "Traffic Sign 20"

[image01]: ./images/random-traffic-signs.png         "Random traffic signs"
[image02]: ./images/training-classes-histogram.png   "Histogram of traffic sign classes from the training dataset"
[image03]: ./images/validation-classes-histogram.png "Histogram of traffic sign classes from the validation dataset"
[image04]: ./images/test-classes-histogram.png       "Histogram of traffic sign classes from the test dataset"

[image05]: ./images/pre-processed.png  "Traffic signs pre-processed"

## Data Set Summary & Exploration

### Dataset size

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 12630
* The size of the validation set is 4410
* The size of test set is 34799
* The shape of a traffic sign image is 32x32x3
* There are 43 unique classes/labels in the data set


### Visualization

In order to visualize the data set I visualized some random signs in order
to get a feel for what the images looked like. Every time the code is run
it shows different images:

![Random Traffic signs][image01]

Also for each of the training, validation and test sets I used histogram to
check the breakdown in various classes of traffic signs:

![Histogram of traffic sign classes from the training dataset][image02]

![Histogram of traffic sign classes from the validation dataset][image03]

![Histogram of traffic sign classes from the test dataset][image04]


## Design and Test a Model Architecture

### Image pre-processing

As a first step, the images are converted to grayscale and some form
of histogram equalization is applied.

The thinking behind that decision was based on the fact that the data
set included images that were showing quite different characteristics:

* dark images with low contrast
* bright images with low contrast
* images with high contrast
* images with a low saturation
* images with a high saturation

The images are pre-processed as follows:

* Convert RGB image to greyscale.
* Histogram equalization using OpenCV's CLAHE functionality (Contrast
  Limited Adaptive Histogram Equalization).

Here are some examples of traffic sign images after grayscaling
and histogram equalization:

![After pre-processing][image05]

As a last step, the image data is normalized in order to bring the range of
pixel values in the -1.0, +1.0 range, and try to get an average that is close
to zero.

This helps with the numerical stability of the algoriths and the convergence
of the learning process.


### Data set augmentation

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


### Model architecture

For the model I started from the LeNet model, and evolved from there until
I got results that met the requirements.

The LeNet model was selected as a basis for my work as it has shown great
success doing classification of greyscale images.

However LeNet had been designed to classify images in 10 classes, where
the traffic sign dataset has 43 classes.

It was quite obvious that some key parameters of the model would have
to be evolved to reach the required classification performance.

The final model consists of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 Greyscale normalized image            |
| Convolution 3x3       | 1x1 stride, same padding, outputs 28x28x108   |
| RELU                  |                                               |
| Average pooling       | 2x2 stride,  outputs 14x14x108                |
| Convolution 3x3       | 1x1 stride, same padding, outputs 10x10x200   |
| RELU                  |                                               |
| Average pooling       | 2x2 stride,  outputs 5x5x200                  |
| Fully connected       | 5000 (result of flattening the previous)      |
| RELU                  |                                               |
| Fully connected       | 800                                           |
| RELU                  |                                               |
| Fully connected       | 400                                           |
| RELU                  |                                               |
| Fully connected       | 200                                           |
| RELU                  |                                               |
| Fully connected       | 43                                            |
| Softmax               |                                               |


### Model training

To train the model, the following were used:

* an Adam optimizer
* a batch size of 32
* for 20 epochs

The final model accuracy results were:

* validation set accuracy of 95.2%
* test set accuracy of 94.9%

To break the glass ceiling sitting at around 90% of accuracy the
following adjustments appear to have been critical:

Getting to this has been quite a learning experience as it took me
hundreds of experiments to start gaining an understanding of which
parameters really drove the model performance.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

## Test the model on new images

### Data collection

As I work in Germany I took the opportunity of my morning and evening walks
to and from work to take some pictures using my cell phone.

As can be noticed below the lighting conditions do vary greatly and I made
sure for nightly shots to take some pictures with flash enabled.

![alt text][sign01] ![alt text][sign02] ![alt text][sign03] ![alt text][sign04] ![alt text][sign05] 

![alt text][sign06] ![alt text][sign07] ![alt text][sign08] ![alt text][sign09] ![alt text][sign10] 

![alt text][sign11] ![alt text][sign12] ![alt text][sign13] ![alt text][sign14] ![alt text][sign15] 

![alt text][sign16] ![alt text][sign17] ![alt text][sign18] ![alt text][sign19] ![alt text][sign20] 

In total I gathered pictures of 20 traffic signs which I cropped and
resized to the expected 32x32x3 image size using Gimp and ImageMagick.

No further processing was done on the images.

**NOTE**: Some of these traffic signs are not falling in any of the 43
traffic signs classes present in the dataset.


### Model predictions

Here are the results of the prediction:

| Image                 | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| Stop Sign             | Stop sign                                     |
| U-turn                | U-turn                                        |
| Yield                 | Yield                                         |
| 100 km/h              | Bumpy Road                                    |
| Slippery Road         | Slippery Road                                 |

The model was able to correctly guess 4 of the 5 traffic signs, which gives
an accuracy of 80%. This compares favorably to the accuracy on the test set
of ...


### Prediction probabilities

The code for making predictions on my final model is located in the 11th
cell of the Ipython notebook.

#### First image

For the first image, the model is relatively sure that this is a stop sign
(probability of 0.6), and the image does contain a stop sign.

The top five soft max probabilities were:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

For the second image ... 

