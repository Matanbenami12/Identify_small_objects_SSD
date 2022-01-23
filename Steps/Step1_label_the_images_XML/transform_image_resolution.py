from PIL import Image
import os
import argparse
import tensorflow as tf


def rescale_images(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+img)


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def main(_):
    flags = tf.app.flags
    flags.DEFINE_string('image_dir',
                        "C:/Users/matan/Desktop/pic/",
                        'Path to the image_dir input')               ## Directory name with all of your images

    flags.DEFINE_integer('image_width', '600',
                        'width image')                               ## Size width image.    Rrecomened  width = 600 
    
    
    flags.DEFINE_integer('image_height', '800',
                        'height image')                              ## Size height image.   Rrecomened  height = 800 
    FLAGS = flags.FLAGS

    path_image_dir = os.path.join(FLAGS.image_dir)
    image_width = FLAGS.image_width
    image_height = FLAGS.image_height
    image = image_width, image_height
    rescale_images(path_image_dir, image)

    print('Successfully transform the images: ')

    del_all_flags(tf.flags.FLAGS)


if __name__ == '__main__':
    tf.app.run()
