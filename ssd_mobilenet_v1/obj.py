#!/usr/bin/env python
# coding: utf-8                                                                                                                                                                                                                    \\\\\\                                                                                                                                                                                                                                                                      \\\\

import numpy as np
import os
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from math import sqrt
import cv2
from datetime import datetime


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# ## Env setup

# This is needed to display the images.
##from IPython import get_ipython
##ipy = get_ipython()
##if ipy is not None:
    ##ipy.run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection moduleS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

def m():
    # What model to download
    cap = cv2.VideoCapture(1)
    MODEL_NAME = 'C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/Warning_graph';


    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('C:/Users/matan/Desktop/object_detection_project/tensorflow1/models/research/object_detection/training', 'object-detection.pbtxt')
    NUM_CLASSES = 1

    # length Warning and focal length
    focal_length=1170
    lenght_warning=0.02
    # length Baby and focal length
    focal_length_Baby=1000
    #focal_length_Baby=1050
    lenght_Baby=0.28
    #lenght_Baby=0.2


    # ## Load a (frozen) Tensorflow model into memory.


    ## send a massage to whatsapp



    ##msg = 'Warning Baby is next to a small object'
    ##msg1='Small object'

    ##driver = driverA
    ##msg_box = driver.find_element_by_class_name('_3uMse')
    ##msg_box.send_keys(msg1)

    #_1Plpp


    msg_w=1
    msg_b=1
    #####
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    # ## Helper code


    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    # # Detection

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    ##PATH_TO_TEST_IMAGES_DIR = 'C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/test_images'
    ##TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    # Size, in inches, of the output images.
    ##IMAGE_SIZE = (12, 8)



    def run_inference_for_single_image(image, graph):
        print("single_image")
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    ##i=0
    ##for image_path in TEST_IMAGE_PATHS:
      ##  print("image show")
        ##image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        ##image_np = load_image_into_numpy_array(image)


    counter = 3
    flag=0
    flag1=0
    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                ret , image_np= cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                     image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)




                for i, b in enumerate(output_dict['detection_boxes'][0]):
                   if output_dict['detection_classes'][i] == 1:
                       if output_dict['detection_scores'][i] > 0.8:
                           flag=1
                           print(output_dict['detection_boxes'][i])
                           print(output_dict['detection_scores'][i])
                           mid_x = (output_dict['detection_boxes'][i][3]+output_dict['detection_boxes'][i][1]) / 2
                           mid_y = (output_dict['detection_boxes'][i][2]+output_dict['detection_boxes'][i][0]) / 2

                           ##lenght_warning_image in pixle
                           lenght_warning_image=(output_dict['detection_boxes'][i][3]-output_dict['detection_boxes'][i][1])*800
                           apx_distance = round(lenght_warning/(((lenght_warning_image)/focal_length)),2)
                           #print(output_dict['detection_boxes'][i][3]-output_dict['detection_boxes'][i][1])
                           #print(apx_distance)




                           cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800), int(mid_y*600-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                           if apx_distance <= 0.5:
                               print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                               cv2.putText(image_np, 'Warning'. format(apx_distance), (int(mid_x * 800),
                                           int(mid_y * 600)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)



                               if(msg_w==1):
                                   msg_w=0
                                   ##driver.find_element_by_class_name('_1U1xa').click()
                                   print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

                       elif output_dict['detection_scores'][i] < 0.7:
                           counter=2

                   elif output_dict['detection_classes'][i] == 1:
                       if output_dict['detection_scores'][i] > 0.4:
                           flag1 = 1
                           mid_x1 = (output_dict['detection_boxes'][i][3] + output_dict['detection_boxes'][i][1]) / 2
                           mid_y1 = (output_dict['detection_boxes'][i][2] + output_dict['detection_boxes'][i][0]) / 2


                           lenght_Baby_image = (output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1]) * 800
                           apx_distance1 = round(lenght_Baby / (((lenght_Baby_image) / focal_length_Baby)), 2)
                           print(output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1])
                           #print(apx_distance1)
                           cv2.putText(image_np, '{}'.format(apx_distance1), (int(mid_x1 * 800), int(mid_y1 * 600)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                #print(flag)
                #print(flag1)
                if flag == 1 and flag1 == 1:
                    if apx_distance < apx_distance1:
                       distance = round(sqrt(((apx_distance1) ** 2) - ((apx_distance) ** 2)),1)

                       cv2.putText(image_np, '{}'.format(distance), (int(0.5 * 800) - 200,
                                     int(0.68 * 600)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 2)

                       if distance < 0.5:
                              if(msg_b == 1):
                                msg_b=0
                               ## msg_box.send_keys(msg)
                                ##driver.find_element_by_class_name('_1U1xa').click()

                    #else:
                     #  distance = round(sqrt(((apx_distance1) ** 2) - ((apx_distance) ** 2)),2)
                      # if distance <= 0.5:
                       #    cv2.putText(image_np, '{}'.format(distance), (int(0.5 * 800) - 200,
                        #               int(0.68 * 600)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 3)
               # print(distance)
                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        ##plt.figure(figsize=IMAGE_SIZE)
        ##plt.imshow(image_np)
        ##plt.savefig(f"C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/test_images/iamge{i}.jpg")
        ##image2 = Image.open(f'C:/Users/matan/Desktop/tensorflow1/models/research/object_detection/test_images/iamge{i}.jpg')
        ##image2.show()
        ##i=i+1




m()



