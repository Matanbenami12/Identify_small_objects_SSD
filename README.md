# Identify_small_objects_SSD
In this project we will be investigating the difficulty in identifying small objects and its distance with our own data set ,by used SSD algorithm. 
We will discuss and compare between SSD_v1 to SSD_v1 with atrous filter 


1.	Collect a few hundred images that contain your object - The bare minimum would be about 100, ideally more like 500+, but, the more images you have, the more tedious step 2 is...
2.	label the images, ideally with a program. LabelImg. This process is basically drawing boxes around your object in an image. The label program automatically will create an XML file that describes the object in the pictures.
3.	Split this data into train/test samples
4.	Generate TF Records from these splits
5.	Setup a .config file for the model of choice 
6.	Train
7.	Export graph from new trained model
8.	Detect custom objects in real time for both SSD and SSD with atrous filter  !
9.	Compare between SSD to SSD with atrous filter 

# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim







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
