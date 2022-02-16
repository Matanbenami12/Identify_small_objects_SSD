# Identify_small_objects_SSD(Windows only)
In this project we will be investigating the difficulty in identifying small objects and its distance with our own data set ,by used SSD algorithm. 
We will discuss and compare between SSD_mobilenet to SSD_resnet 

To begin, you are going to make sure you have GPU version of TensorFlow(.1) and all of the dependencies 
If you need to install GPU TensorFlow we recommend to follow this video How to [installTensorFlow_GPU](https://www.youtube.com/watch?v=r7-WPbx8VuY&ab_channel=sentdex
) 
- [ ] 1. Make sure you are install TensorFlow_GPU version 1.14  and you are using python version 3.6 .


Ones you are install GPU TensorFlow the other Python dependencies are covered with

```
pip install jupyter
pip install matplotlib
pip install pillow
pip install lxml
pip install numpy
pip install opencv-python
```


Next, we need to clone the github.
click the green "clone or download" button on the https://github.com/tensorflow/models page, download the .zip, and extract it.


Once you have the models directory navigate to that directory in your terminal/cmd.exe. 

Run dose 3 commaands

```
1. export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
##### Navigate to models/research and run dose commaand

```
2. python setup.py build
   python setup.py install
```

##### Navigate to models/research/slim
Remove ` bulid `and run again dose commaand from slim directory:
```
3. python setup.py build
   python setup.py install

```

Once we finish 🎉 we are ready to make our own data set.

OKKKKK-so a brief overview of what we needed to do 


1.	Collect a few hundred images that contain your object - The bare minimum would be about 100, ideally more like 500+, but, the more images you have, the more tedious step 2 is...
2.	label the images, ideally with a program. LabelImg. This process is basically drawing boxes around your object in an image. The label program automatically will create an XML file that describes the object in the pictures.
3.	Split this data into train/test samples
4.	Generate TF Records from these splits
5.	Setup a .config file for the model of choice 
6.	Train
7.	Export graph from new trained model
8.	Detect custom objects in real time for both SSD_mobilenet and SSD_resnet  !
9.	Compare between SSD_mobilenet to SSD_resnet 

So let's begin.

We are going to cover how to create the TFRecord files that we need to train an object detection model.

Go to derctory calld [`Steps`](https://github.com/Matanbenami12/Identify_small_objects_SSD/tree/main/Steps) and make sure you are create the required TFRecord files from those steps.
 
 
After we make our own TFRecord files we are ready to tarinin our model,


We will train our object detection model to detect our custom object. To do this, we need the Images, matching TFRecords for the training and testing data, and then we need to setup the configuration of the model, then we can train.  
That means we need to setup a configuration file.

We can use a pre-trained model, and then use transfer learning to learn a new object,

TensorFlow has quite a few pre-trained models with checkpoint files available, along with configuration filesfrom [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
)

We are going to go with SSD mobilenet, using the following checkpoint
 http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
 and [configuration]( https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config
) file






###  What we do now is the same for both SSD_mobilenet and SSD_resnet 

We make a new directory calld training and  Put the config in , and extract the ssd_mobilenet_v1 in the models directory

In the configuration file, you need to search for all of the PATH_TO_BE_CONFIGURED points and change them. You may also want to modify batch size. Currently, it is set to 24 in my configuration file. Other models may have different batch sizes. 
Finally, you also need to change the checkpoint name/path, num_classes to 1, num_examples to 12, and label_map_path: "***Your_PATH***/models/research/object_detection/training/object-detect.pbtxt"



Inside training dir, add object-detection.pbtxt:
``` ruby
item {
  id: 1
  name: 'Warning'
}
```

Like that:


![צילום מסך 2022-01-26 073509](https://user-images.githubusercontent.com/56115477/151109540-f7ff83d6-9a8b-42ca-9c0d-59b393cd45ba.png)





 All directories **training**, **data**, **images**, **ssd_mobilenet_v1_coco_11** 
 move directly into the ***Your_PATH***\models\research\object_detection directory. 




 At this point, here is what your \object_detection folder should look like:

![3](https://user-images.githubusercontent.com/56115477/151109553-2dabe134-75b8-434e-bdc1-0c96cc9a0c55.png)







***Navigate to research/object_detection/legacy/train.py***

You need to change the  ***train_dir path and pipeline_config_path*** like:


```
flags.DEFINE_string('train_dir', "C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/training/",
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', "C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/training/ssd_mobilenet_v1_pets.config",
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

```
And now, the moment of truth! run in your terminal/cmd.exe this command
python train.py


Barring errors, you should see output like:


![step](https://user-images.githubusercontent.com/56115477/151669704-d6616676-ee09-43f6-aaaa-954cd286be5f.png)



Your steps start at 1 and the loss will be much higher. You want to shoot for a loss of about ~1 on average (or lower). We wouldn't stop training until you are for sure under 1. You can check how the model is doing via TensorBoard. Your models/research/object_detection/training directory will have new event files that can be viewed via TensorBoard.

From models/research//object_detection, via terminal, you open cmd.exe and start TensorBoard with:

tensorboard --logdir="C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/training"  --host localhost 

This runs on 127.0.0.1:6006 (visit in your browser)

 TensorBoard graph result are [here](https://github.com/Matanbenami12/Identify_small_objects_SSD/tree/main/TensorBoard_result)

Now,we're going to export the graph and then test the model.

In the models/research/object_detection/training" directory, there is a script for us: export_inference_graph.py

To run this, you just need to pass in your checkpoint and your pipeline config, then wherever you want the inference graph to be placed. 

Your checkpoint files should be in the training directory. Just look for the one with the largest step (the largest number after the dash), and that's the one you want to use. Next, make sure the pipeline_config_path is set to whatever config file you chose, and then finally choose the name for the output directory,We went with Warning_graph

Run the command from models/research/object_detection
```
python export_inference_graph.py  --input_type image_tensor --pipeline_config_path C:/Users/matan/Desktop/ object_detection_project/tensorflow1/models/research/object_detection/training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix C:/Users/matan/Desktop/ object_detection_project/tensorflow1/models/research/object_detection/training/ model.ckpt-80638--output_directory Warning_graph
```

Now, you should have a new directory, in our case,  is Warning_graph, inside it, We have new checkpoint data, a saved_model directory, and, most importantly, the forzen_inference_graph.pb file.

Now, we're just going to run the sample obs.py


### What model to download.
```ruby
MODEL_NAME = 'C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/Warning_graph';
```
### Path to frozen detection graph. This is the actual model that is used for the object detection.
```ruby
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
```
### List of the strings that is used to add correct label for each box.
```ruby
PATH_TO_LABELS = os.path.join('C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/training', 'object-detection.pbtxt')

NUM_CLASSES = 1

```

We will do all again with SSD resnet
and compare between SSD_mobilenet to SSD_resnet 
 
 
object detection in real-time

 Once we identify small objects in image, we decides to move forward and identify object and  distance from camera in real-time
 It's a few edits 
 So, here, we can iterate through the boxes for further analysis. Boxes are an array, inside of an array, so, to iterate through them, we need to do:
 
       for i,b in enumerate(boxes[0]):
Now, for the "too close" tracking, we're interested in some specific classes. One could argue that *any* object that is too close is something we might want to avoid. The deal is, however, that, to determine distance, you need to know the object's size before-hand.

For example, a car that is too close, being a few feet across if it's driving in front of you, will be much larger than a tree trunk, even if they're the exact distance from you. For this reason, we're only going to lump in cars, buses, and trucks for this loop. You can detect other smaller or larger objects in other loops if you like.

      for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
Next, we want to be fairly certain these are *actually* vehicles. For example, in the vis_util.visualize_boxes_and_labels_on_image_array function, the default parameter for drawing boxes is a score of 0.5. I discovered this by simply viewing the code to see what all of the options were. We can use the same score of 0.5 or more logic. It's important to note that the object detector actually detects quite a few more objects, you just might not have been aware since only the scores of 0.5 or greater were being drawn.

      for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
          if scores[0][i] >= 0.5:
Now, we want to measure the width of the detected object. We can do this by asking how many pixels-wide the object is. We can do this with:

apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))),1)
For some added granularity, I am going to add:

apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
You can play with this calculation more if you like, but I am going to move on. For debugging purposes, I would like to display this number on screen. To do this, I am going to display at the following coordinates:

            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
Basically, this displays at left of the detected object, in the middle vertically.

      for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
          if scores[0][i] >= 0.5:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
We can write this to the screen with:

            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
If that distance is less than 0.5, then I am going to say this is too close for an object ~the width of a car, so we want to display a warning:

            if apx_distance <=0.5:
              if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
Also note: if mid_x > 0.3 and mid_x < 0.7:. This code just looks to see that, even if the object is close, it might be far enough to the side that it's not a likely collision issue. Obviously, to do this right, you would probably want to track object history and trajectory. For example, if you're going through an intersection, and there's a car coming across, a reasonable person would know that, despite the car not being in front, that vehicle is a collision risk.

Full code up to this point:
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 We are going to cover how to found distance from camera to object [here](https://github.com/Matanbenami12/Identify_small_objects_SSD/tree/main/Distance%20from%20camera) 
 using  OpenCV package
 
 1. we will found distance from camera to object using  OpenCV package
 In order to determine the distance from our camera to a known object   

Now let us see the diffrent

Real time identify:


| Model name     | Traning time  | real time Speed (ms) | real time Identify 0.5m|  real time Identify 1m| real time Identify 1.5m|   
| -------------  | ------------- | -------------------- | --------------------   | --------------------  | --------------------   |
| SSD_mobilenet  |     52 hours  |           30         |                        |                       |                        |
| SSD_resnet     |     25 hours  |           237        |                        |                       |                        |




Images identify:
  
  
|   Identify_small_objects|    |   | 
|--|--|--|            
|  SSD_mobilenet | TOP 1 |  TOP 5 |  
|     0.5 m      |       |        |
|      1 m       |       |        |
|     1.5 m      |       |        |
|   SSD_resnet   | TOP 1 |  TOP 5 |              
|     0.5 m      |       |        |
|      1 m       |       |        |
|     1.5 m      |       |        | 
