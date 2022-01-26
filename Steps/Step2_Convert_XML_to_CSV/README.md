Once you have over 100+ images labeled, we're going to separate them into training and testing groups. To do this, just copy about 10% of your images and their annotation XML files to a new dir called test and then copy the remaining ones to a new dir called train

Now we need to convert these XML files to singular CSV files that can be then converted to the TFRecord files. To do this, I am going to make use of some of the code from  https://www.kaggle.com/sanjaydogood/xml-to-csv-py, with some minor changes. To begin, we're going to use xml_to_csv.py. You can either clone his entire directory(
Step2_Convert_XML_to_CSV) or just grab the files, we'll be using two of them.
Within the xml_to_csv script, I changed:

```ruby
def xml_to_csv_utility():
    os.chdir("/kaggle/input/face-data/homo_encrypt/image_data")
    for image_set in ['train_data','test_data']:
        image_path = os.path.join(os.getcwd(), image_set)
        print(image_path)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('{}.csv'.format(image_set.split("_")[0]), index=None)
        print('Successfully converted xml to csv.')
```        
        
To:
```ruby
def main():
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(), 'C:/Users/matan/Desktop/object_detection_project/images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv("C:/Users/matan/Desktop/object_detection_project/data/{}_labels.csv" .format(directory) ,index=None)
        print('Successfully converted xml to csv.')


main()
```


This just handles for the train/test split and naming the files something useful. Go ahead and make a data directory, and run this to create the two files.. At this point, you should have the following structure:



![image](https://user-images.githubusercontent.com/56115477/149844500-d008a6ff-f4a4-47c0-978e-2f5fe2b82348.png)




You finish......... 
Just kidding we are so close to the end of those steps well done go the next step(Step3_Generate_TFRecord)

