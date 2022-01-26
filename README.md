# Identify_small_objects_SSD(Windows only)
In this project we will be investigating the difficulty in identifying small objects and its distance with our own data set ,by used SSD algorithm. 
We will discuss and compare between SSD_v1 to SSD_v1 with atrous filter 

To begin, you are going to make sure you have GPU version of TensorFlow(.1) and all of the dependencies 
If you need to install GPU TensorFlow we recommend to follow this video (How to installTensorFlow_GPU )
https://www.youtube.com/watch?v=r7-WPbx8VuY&ab_channel=sentdex
- [ ] 1. Make sure you are install TensorFlow_GPU version 1.14  and you are using python version 3.7  .


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
8.	Detect custom objects in real time for both SSD and SSD with atrous filter  !
9.	Compare between SSD to SSD with atrous filter 

So let's begin.

We are going to cover how to create the TFRecord files that we need to train an object detection model.

Go to derctory calld `Steps` and make sure you are create the required TFRecord files from those steps.
 
 
After we make our own TFRecord files we are ready to tarinin our model,


we will train our object detection model to detect our custom object. To do this, we need the Images, matching TFRecords for the training and testing data, and then we need to setup the configuration of the model, then we can train.  
That means we need to setup a configuration file.

We can use a pre-trained model, and then use transfer learning to learn a new object,

TensorFlow has quite a few pre-trained models with checkpoint files available, along with configuration filesfrom here

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
I am going to go with SSD mobilenet, using the following checkpoint and configuration file


 https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config
 http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11

and

https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11


### All what we do now is the same for both ssd_v1 and ssd_v1_artous

We make a new directory calld training and  Put the config in , and extract the ssd_mobilenet_v1 in the models directory

In the configuration file, you need to search for all of the PATH_TO_BE_CONFIGURED points and change them. You may also want to modify batch size. Currently, it is set to 24 in my configuration file. Other models may have different batch sizes. 
Finally, you also need to change the checkpoint name/path, num_classes to 1, num_examples to 12, and label_map_path: "training/object-detect.pbtxt"

It's a few edits, so here is my full configuration file:


Inside training dir, add object-detection.pbtxt:
``` ruby
item {
  id: 1
  name: 'Warning'
}
```
 All directories **training**, **data**, **images**, **ssd_mobilenet_v1_coco_11** move directly into the C:\tensorflow1\models\research\object_detection directory. 

At this point, here is what your \object_detection folder should look like:
And now, the moment of truth! From within models/object_detection:

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

Barring errors, you should see output like:

INFO:tensorflow:global step 11788: loss = 0.6717 (0.398 sec/step)
INFO:tensorflow:global step 11789: loss = 0.5310 (0.436 sec/step)
INFO:tensorflow:global step 11790: loss = 0.6614 (0.405 sec/step)
INFO:tensorflow:global step 11791: loss = 0.7758 (0.460 sec/step)
INFO:tensorflow:global step 11792: loss = 0.7164 (0.378 sec/step)
INFO:tensorflow:global step 11793: loss = 0.8096 (0.393 sec/step)
Your steps start at 1 and the loss will be much higher. Depending on your GPU and how much training data you have, this process will take varying amounts of time. On something like a 1080ti, it should take only about an hour or so. If you have a lot of training data, it might take much longer. You want to shoot for a loss of about ~1 on average (or lower). I wouldn't stop training until you are for sure under 2. You can check how the model is doing via TensorBoard. Your models/object_detection/training directory will have new event files that can be viewed via TensorBoard.

From models/object_detection, via terminal, you start TensorBoard with:

tensorboard --logdir='training'

This runs on 127.0.0.1:6006 (visit in your browser)

My total loss graph:




***Navigate to research/object_detection/lagcy/train.py*** that file in your terminal/cmd.exe









I am doing this tutorial on a fresh machine to be certain I don't miss any steps, so I will be fully setting up the Object API. If you've already cloned and setup, feel free to skip the initial steps and pick back up on the setup.py part!

First, I am cloning the repository to my desktop:

git clone https://github.com/tensorflow/models.git

Then, following the installation instructions:

sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib
And then:

# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
If you get an error on the protoc command on Ubuntu, check the version you are running with protoc --version, if it's not the latest version, you might want to update. As of my writing of this, we're using 3.4.0. In order to update or get protoc, head to the protoc releases page. Download the python version, extract, navigate into the directory and then do:

sudo ./configure
sudo make check
sudo make install
After that, try the protoc command again (again, make sure you are issuing this from the models dir).





python train.py --logtostderr --train_dir=C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/training/ --pipeline_config_path=C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/training/ssd_mobilenet_v1_pets.config





if its say no named objet_detction
go to C:/Users/avi/Desktop/tensorflow1/models/research
and put:
setup.py build
setup.py install


if it sat no named nets 
go to C:/Users/avi/Desktop/tensorflow1/models/research/slim
remove bulid
and put:
python setup.py build
python setup.py install
