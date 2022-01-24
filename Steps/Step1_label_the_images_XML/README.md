# Make XML file


We used our on data, to make some images of small objects
We take the marbles for the purpose of illustration:
Take different types of size / color marbles.
The marbles we photographed in the background with sunlight, shade and house light.
In these backgrounds we photographed the marbles on different places dirt, grass, floor etc.

In general, change the pictures around the size of 800x600 with transform_image_resolution.py 
When running this:

Pick the directory with all of your images and choose the size image (width ,height ) thet u want,
we recomened to choose size  800x600  not too large and not too small.


For this step , you can track anything you want, you just need 100+ images. Once you have images, you need to annotate them.
I am going to use LabelImg(LabelImg is a graphical image annotation tool).


Installation instructions:

U only need to download directory LabelI and click on labelImg.exe( only for windows for others go here https://github.com/tzutalin/labelImg.): 


When running this, you should get a GUI window. From here, choose to open dir and pick the directory that you saved all of your images to. Now, you can begin to annotate with the create rectbox button. Draw your box, add the name in, and hit ok. Save, hit next image, and repeat! You can press the w key to draw the box and do ctrl+s to save faster. Not sure if there's a shortcut for the next image.
![image](https://user-images.githubusercontent.com/56115477/149636550-9f5c8b3f-1624-4585-a0b1-84abb072b006.png)


Once you LabelImg all the image i can say to you: 

great job, now u ready for next step (Step2_Convert_XML_to_CSV).
