
Once you've done all of this,we're going to cover how we can create the required TFRecord files from this data.
Now, just grab the files generate_tfrecord.py from here https://blog.roboflow.com/create-tfrecord/ and the utils directory. The only modification that you will need to make here is in the class_text_to_int function and , patch of csv_input,Path to output TFRecord ,Path to images .
You need to change this to your specific class. In our case, we just have ONE class and Path.

# Path:
```ruby
flags = tf.app.flags
flags.DEFINE_string('csv_input', "C:/Users/matan/Desktop/object_detection_project/data/{}_labels.csv".format(directory),
                'Path to the CSV input')
flags.DEFINE_string('output_path', "C:/Users/matan/Desktop/object_detection_project/data/{}.record".format(directory),
                'Path to output TFRecord')
flags.DEFINE_string('image_dir', "C:/Users/matan/Desktop/object_detection_project/images/{}".format(directory),
                'Path to images')
FLAGS = flags.FLAGS
```




# Class:
If you had many classes, then you would need to keep building out this if statement.

```ruby
def class_text_to_int(row_label):
    if row_label == 'Warning':
        return 1
    else:
        None
```

Next, in order to use this, we need to either be running from within the models directory of the cloned models github:
https://github.com/tensorflow/models.git


Or we can grab the utils directory
Choose what u want we choose to grab the utils directory



Now, in your data directory, you should have train.record and test.record.



