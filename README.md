# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram1]: ./writeup_images/histogram1.jpg "Set counts"
[image2]: ./writeup_images/color1.jpg "Color image"
[image3]: ./writeup_images/grayscale1.jpg "Grayscale image"
[image4]: ./writeup_images/5_new_signs.jpg "The 5 new signs"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KenOverholt/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic Python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the test set is 12630
* The size of the validation set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the count of data in each of the train, test, and validation sets.  Additionally, I display one of the images from the original, color image set but I'll include that later in this discussion.

![alt text][histogram1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the example paper provided used grayscale.  It seems to be quicker to train since there are fewer color values.  While color can make signs stand out and add additional queues as to a signs content, the images, characters, and drawings on each sign is generally sufficient to identify a sign.  Color is generally an extra indication and maybe a general classification a sign category.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]![alt text][image3]

As a last step, I normalized the image data because the network does a better job of analyzing data in the -1 to 1 range as opposed to larger values.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For my final model, I used the LeNet architecture modified for the grayscale images.  It consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale PNG image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		   |   									|
| Max pooling				    | 2x2 stride, valid padding, outputs 5x5x16        									|
|	Flatten					|	outputs 400											|
|	Fully connected					| outputs 120												|
| RELU      |         |
| Fully connected     | outputs 84 |
| RELU      |         |
| Fullly connected    | outputs 43 |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the standard LeNet parameters that we had for our lab but I increased the epochs to 30.  I found that converting to grayscale, normalizing, and increasin the epoch count gained enough performace to move above the required .93 mark.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.934
* test set accuracy of 0.911

I chose a well-known architecture:
* What architecture was chosen? I chose the LeNet architecture provided in the lab since I was running out of time to complete the project.  I would love to have modified the architecture but just ran out of time.  I spent time exploring various grayscale conversion techniques and normalization methods.  I discarded ones that dropped the accuracy and kept the one of each that increased accuracy.
* Why did you believe it would be relevant to the traffic sign application?  The architecture learns in stages, first learning the smallest, generic elements.  Then it combines them to larger pieces and so-on until it assembles the final image.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  My model training stopped with a validation of 0.934 accuracy.  Running the test against the model gave a similar result of only .023 difference indicating that that model does not overfit the data to much.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web along with their preprocessed version (normalized, grayscale) to their right:

![alt text][image4]

1. The first image should be easy to classify.  It has an even, contrasting background.  It has a relatively straight-on view.  It has a clear, large 50 on it which is difficult to mistake for other sign types.
2. The second image is more difficult to classify.  It's background is rough including many tree branches interspersed with sky.  The image on the sign includes two people which can be difficult to make out.
3. The third sign seems to be in the middle of calssification difficulty.  It's background starts a bit dark at the top and brightes up as it moves down.  Then there are dark green trees at the bottom.  The sign only takes up the left half of the image so it is smaller giving it fewer pixels to for an image.  Additionally, it is really two signs.  One includes arrows in a circle (the roundabout) and above it is a type of yield sign.  It is possible that the training images do not include the yield part and only include the circle part.
4. The fourth sign is a right-turn sign and again, it is smaller and there is a second, smaller right-turn in the background.  The remaining background is also busy with a wall and greenery.
5. The fifth sign is easy to recognize as STOP is printed across it.  The background varies a bit but is much lighter than the dark red background of the STOP sign.  This sign also takes up most of the frame leaving only a little bit of background. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/hr      		| 50 km/hr   									| 
| Pedestrians     			| 	Roadwork									|
| Roundabout					| 	50 km/hr								|
| Right-turn	      		| 	Priority Road				 				|
| Stop Sign			| Stop Sign      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. While this result is much lower than my training, test, and validation sets, these signs are not all in the most favorable conditions.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 50 km/hr   									| 
| 0.99     				| U-turn 										|
| 0.99					| Yield											|
| 0.77	      			| Priority Road					 				|
| 1.00				    | Stop sign      							|

* The second image was way offbase as the actual value wasn't even in the top 5 choices.
* The third sign actually included 2 signs but since they were on the same pole and related and since I don't know how German signs work, I assumed they part of the same "sign".  The model predicted the sign as a yield sign with almost 100% accuracy and the top of two signs is a yield sign so this should probably be counted as accurate.
* The fourth sign was rather high in certainty but was not accurate and one of the top 5 choices were accurate.
* The first and fifth signs were 100% certain and those were the most obvious signs.

Note that the html version included in my submission includes the trained model which I used for the data from the 5 signs above.  Later found I had missed a couple of items for the writeup so I trained it one more time after adding in a couple cells to get the accuracy of the test and validation sets.  The actual Jupyter notebook includes those code cells near the end and it includes different 5-sign results.  The 1st and 5th sign are still predicted accurately but the roundabout is now accurate.  It predicted the sign as a roundabout instead of a yield.  It's accuracy also pretty high as 0.852.  It is interesting that yield, which was predicted in the last model, wasn't even in the top 5 softmax probabilities this time.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


