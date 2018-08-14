# **Behavioural Cloning** 

---

###Behavioural Cloning Project

**The goals / steps of this project are the following:**

* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarise the results with a written report


[//]: # (Image References)

[image1]: ./output_data/Final_not_elu_Loss_NVidia_8.png "RELU loss 8 epoch"
[image2]: ./output_data/Final_old_data_elu_Loss_NVidia_8.png "ELU loss 8 epoch"
[image3]: ./output_data/Final_Loss_NVidia_6.png "ELU loss 6 epoch"
[image4]: ./output_data/Loss_NVidia_6.png "RELU loss 6 epoch - final model"
[image5]: ./output_data/DirtTrack.jpg "Dirt Track Veering"
[image6]: ./output_data/bridge.jpg "Bridge Weaving"
[image7]: ./output_data/udacity_compare.png "DataSet Comparison"
[image8]: ./output_data/NeuralNetDiagram.001.png "Net diagram"
[image9]: ./output_data/left_camera.jpg "Left Camera"
[image10]: ./output_data/centre_camera.jpg "Centre Camera"
[image11]: ./output_data/right_camera.jpg "Right Camera"
[image12]: ./output_data/centre_camera_cropped.jpg "Cropped Image"






---
### Files Submitted & Code Quality

#### 1. The project includes the following files:
* [model.py](./model.py) - python file containing the script to create and train the model
* [drive.py](./drive.py) - python file for driving the car in autonomous mode
* [model.h5](./model.h5) - python file of the trained convolution neural network 
* [video.mp4](./video.mp4) - video recording of the model driving vehicle in autonomous mode
* [writeup report](./my_writeup_template.md) - markdown file summarising the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and the included drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The code is fully commented to explain its operation.

The file structure is made up of 5 functions:

* __main__ - this function is entered when running the python file. It executes each of the other 4 functions in the order shown. An explanation of each function is added below.
    

``` python  
    
    dataset = get_file_data('./my_driving')

    X_train, y_train = prepare_data(dataset)

    model = net_NVIDIA()
    num_epoch = 6

    history_object = train_model(model, X_train, y_train, './NVidia_', num_epoch)

```    

* **get_file\_data** - opens the csv file and loads each line of file into an array/list
* **prepare_data** - extracts each line from csv list and utilises the information to build image array containing left, right and centre camera images, and another array containing the steering angles. The steering angles are compensated for the left and right images by applying a correction factor based on the centre steering angle.
* **net_NVIDIA** - this is the neural network - it is based off the NVIDIA model. This function creates an instance of the model ready for training.
* **train_model** - this function takes in the model and arrays; splits the data so some can be use for validation; trains the model and saves the output.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with both 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 101-121).
The summary of the model is as shown in the table:


| Layer             | Description                                   | Param   |
| -----------------:|:--------------------------------------------- | ------- |
| Input             | 160x320x3 RGB image                           |         |
| lambda_1          | normalise & mean centre each element          | 0       |
| cropping2d_1      | crop image - outputs 65x320x3 image           | 0       |
| conv2d_1          | 5x5 convolution, 2x2 stride, outputs 31x158x24| 1824    |
| RELU              | Activation layer                              |         |
| conv2d_2          | 5x5 convolution, 2x2 stride, outputs 14x77x36 | 21636   |
| RELU              | Activation layer                              |         |
| conv2d_3          | 5x5 convolution, 2x2 stride, outputs 5x37x48  | 43248   |
| RELU              | Activation layer                              |         |
| conv2d_4          | 3x3 convolution, 1x1 stride, outputs 3x35x64  | 27712   |
| RELU              | Activation layer                              |         |
| conv2d_5          | 3x3 convolution, 1x1 stride, outputs 1x33x64  | 36928   |
| RELU              | Activation layer                              |         |
| flatten_1         | Output 1D shape                               | 0       |
| dense_1           | fully connected 2112 inputs, 100 outputs      | 211300  |
| dense_2           | fully connected 100 inputs, 50 outputs        | 5050    |
| dense_3           | fully connected 50 inputs, 10 outputs         | 510     |
| dense_4           | fully connected 10 inputs, 1 output           | 11      |


Total Trainable parameters: 348,219

The model includes RELU layers to introduce nonlinearity (code line 101, 104, 107, 109, 111), and the data is normalised in the model using a Keras lambda layer (code line 91). 

I experimented with 'ELU' based on a recommendation from: [Which Activation](https://jovianlin.io/which-activation-function)


**Listed in the order of recommendation:**

``` html

    • Exponential Linear Unit (ELU): keras.layers.ELU(alpha=1.0)
    • Leaky ReLU + variants: keras.layers.LeakyReLU(alpha=0.01)
            •Randomized ReLU (RReLU)
            •Parametric Leaky ReLU (PReLU)
    • ReLU
    • tanh
    • sigmoid/logistic
```

However with this simple exchange of activations the simulator result was very bad, the vehicle wanted to drive anywhere but on the road.

#### 2. Attempts to reduce overfitting in the model

The model **does not** contain any dropout layers in order to reduce overfitting. I experimented with dropout function in the convolutional layers to try to push the number of epoch, but the model output did not generate good results on the simulator.

I built a number of models, with low success rates. The model would start to overfit quite quickly normally around 5 - 7 epoch. Depending on how the data was prepared this could occur even quicker.


![alt text][image1]  
**Above figure:** With the ReLU and dataset(x) the overfitting occurs very quickly:  

![alt text][image3] ![alt text][image2]  
**Left figure:** Employing an ELU activation function using dataset(x) the overfitting still occurs.  
**Right figure:** Model is identical, simple change to dataset(y) and the overfitting did not occur - however the model always tried to drive off the road in the simulator.

![alt text][image4]  
**Above figure:** The final model with a ReLU activation and dataset(z). The model started to overfit from epoch 7. 

In order to reach this final point the model was trained, validated and tested in the simulator using numerous datasets. With the early stages I had a low success in getting the vehicle to go around the track; This was mostly influenced by the quality of the data collected. I will discuss this point further in **appropriate training data.**

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

#### 4. Appropriate training data  

Getting the most appropriate data proved to be the biggest challenge with this project. I was using a mouse pad to steer the vehicle in the simulator as I couldn't get access to a joypad. The simulator movement around the track appeared smooth as I navigated around the track.

In the early stages evaluated some models with the Udacity dataset and obtained instantly better results. I did a lot of investigation into the dataset to understand where I was going wrong - this is discussed more in the detail of creation data.

Some analysis of my data versus Udacity data showed:

![alt text][image7] 

The udacity dataset appears to have a wider variation and more dense concentration of steering values during the course manoeuvres when compared with my dataset. This could be due to operating the simulator with my mouse pad.

Evaluating the split of the data in both cases: (6000 samples of each)

**Udacity:**  
Total number of steering instances: 6000  
Number of instances with 0 as steering Angle: 3278 (54.63%)  
Number of instances < +/-1 degree as steering Angle: 3396 (56.60%)  
Number of instances with left steering Angle: 1049 (17.48%)  
Number of instances with right steering Angle: 1673 (27.88%)  

**My Dataset:**  
Total number of steering instances: 6000  
Number of instances with 0 as steering Angle: 81 (1.35%)  
Number of instances < +/-1 degree as steering Angle: 1602 (26.70%)  
Number of instances with left steering Angle: 4649 (77.48%)  
Number of instances with right steering Angle: 1270 (21.17%)  

My dataset appeared to not have many 0 degree steering angles when compared with Udacity dataset, but a large jump to small angles +/- 1 degree from 0. This was due to my method of driving in the simulator in that I always had the mouse pad pressed and therefore a small reading was obtained. I tried driving by releasing mouse pad on the straights of the course and this did created areas where 0 degrees existed, however the trained model driving did not improve so I did not decide to use that data collection method.

A summary of my final data collection is shown below:

**My Final Data set:**  
Total number of steering instances: 38229  
Total number of image instances: 38229  
Number of instances with 0 as steering Angle: 1073 (2.81%)  
Number of instances < +/-1 degree as steering Angle: 4038 (10.56%)  
Number of instances with left steering Angle: 20456 (53.51%)  
Number of instances with right steering Angle: 16700 (43.68%)  

The data was collected to try to equalise the number of left and right turns between the instances. This was achieved by using all 3 cameras, driving in both directions of the track and recording data driving around bends in different directions and manners was to augment the dataset with bends experience.

At the bridge challenge the car was snaking across the bridge, and I wanted it to drive across in a straighter line. I tried to append data of driving straight over the bridge a number of times in both directions, but this had a negative impact on the rest of the course, and caused a bridge crash every time.
    
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilise the NVIDIA convolutional model structure. I read their paper on this model and their success. I did try employing some of the things they mentioned such as ELU and YUV image conversion. I did not utilise these in the final model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The split I I eventually used was 80:20; this gave the following number of samples for each:  

**Train on 30583 samples, validate on 7646 samples**


I had various complications with overfitting and tried dropout, other activations, but did not get the results in the simulator. The best solution was to increase the quality of the dataset and limit the epoch to 6 and this produced the following result:

![alt text][image4]  
**Above figure:** The final model with a ReLU activation.

During my experiments I was unhappy about the vehicle navigating the course at 9mph, so increased the speed in the drive.py file. At the high speed of 30mph, none of my models would have any success.
I was having some success at 15mph but one small area of the course the vehicle would clip the edge of the road; lowering the speed to 12mph the vehicle could autonomously navigate the course successfully.

The neural net only takes in the steering angles and does not consider speed when training. Adding the speed component into the behaviour cloning would have allowed the vehicle to manage and learn the manoeuvres at varying speeds and therefore would have improved the quality of its autonomous drive.


#### 2. Final Model Architecture

The final model architecture (model.py lines 82-126) is a convolution neural network the layers and layer sizes are shown in the visualisation of the architecture:

![alt text][image8]

#### 3. Creation of the Training Set & Training Process

As previously discussed I had a lot of issue early on trying to get the vehicle to stay on the track. I eventually tried training based on the Udacity provided data. The car stayed on the track and went around the complete track.

My initial data collection was:
* Drive one lap, save data.
* Drive one lap in reverse direction, save data.
* Drive around course meandering to try to get car back to centre, save data

To try to understand why my data and the Udacity data were giving such different results I compiled a video of my dataset and the Udacity data set to understand the different data collection strategy. The function video_maker was adapted form the video.py supplied by Udacity to take a dataset of images and compile them into a video.

```python

    from moviepy.editor import ImageSequenceClip
    import argparse
    import os

    IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


    def video_maker(image_folder='./data/IMG',set_fps=10):

        #convert file folder into list filtered for image file types
        image_list = sorted([os.path.join(image_folder, image_file)
                            for image_file in os.listdir(image_folder)])
        
        image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

        #two methods of naming output video to handle varying environments
        video_file_1 = image_folder + '.mp4'
        video_file_2 = image_folder + 'output_video.mp4'

        print("Creating video {}, FPS={}".format(image_folder, set_fps))
        clip = ImageSequenceClip(image_list, fps=set_fps)
        
        try:
            clip.write_videofile(video_file_1)
        except:
            clip.write_videofile(video_file_2)

```

I could easily visualise the Udacity data was simply 3 continuous laps of the track in the same direction.

My next data collection was:
* Drive three laps, save data.
* Drive three lap in reverse direction, save data.

After training with my model I could stay on the track on the straights but at the bends the car would veer off-road and would start driving in that area.

Two significantly challenging areas were:

![alt text][image5]     ![alt text][image6]  

**left**: dirt track - the vehicle would veer off between the two posts and start to use the dirt track as the road- it would then successfully navigate back to the road at the other end.  
**right**: bridge - the vehicle would crash into the end or into the sides of the bridge.

In order to navigate around the bends, I appended more training data to the dataset. At the dirt-track bend I drove the vehicle to just before the bend and then set recording of the data:  

* drive around bend in normal direction, save
* Drive bend in reverse direction, save
* drive around bend in normal direction, save
* Drive bend in reverse direction, save

after another retrain of the model the vehicle successfully navigate the whole track at 9mph.

My final data collection:

* Drive three laps with centre land driving, save data.
* Drive three laps with centre land driving in reverse direction, save data.
* Drive around bend in normal direction, save (number of times)
* Drive bend in reverse direction, save (number of times)

The data collected was for left, right, and centre cameras. Below is a sample of these images:

![alt text][image9] ![alt text][image10] ![alt text][image11]

To capture good driving behaviour, I recorded three laps on track one using centre lane driving. As the course is continually turning left, I drove the course in the opposite direction for three laps with good driving behaviour. This resulted in approximately equalised left and right turning in the dataset, in an attempt to avoid the model being biased to one direction of turning.

In order to improve the models ability to go around bends, I went to different bends in the course. The first two data collections were good driving behaviour for just the bend in both directions. I further augmented this by adding recovery driving, by taking the bend wide, or sharply. I varied the speed while going round the bend; slowing at approach then accelerating around it in a smooth manner.

In the neural network the images are cropped to reduce the amount of data requiring processing. 70 pixel rows were removed from top, and 25 pixel rows were removed from the bottom. This data was deemed unnecessary for the model.


![alt text][image10]     ![alt text][image12]   
**Left figure:** Original Image see by centre camera  
**Right figure:** Cropped image of centre camera fed to network


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by discussions in the __Attempts to reduce overfitting in the model__.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

As can be seen in the video.py the vehicle drives around the track and does not leave the drive-able portion.
