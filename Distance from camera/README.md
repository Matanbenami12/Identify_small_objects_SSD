
In order to determine the distance from our camera to a known object , we are going to use Gaussian lens formula with few edits for our project .
First we will look at the following figure to understand Gaussian  lens formula: 
 
![image](https://user-images.githubusercontent.com/56115477/154123479-3e4ba37a-54d8-4f5d-81f7-894f132fabc6.png)
  
  
Given an converging lens, MN  straight line passing through the geometrical centre of a lens and joining the two centres of curvature of its surfaces passing through the point O 
Source light A.
F_1 and F_2 are the focal points of the lens. 

The image formed by a thin lens can be located by drawing three Kepler rays: (1) a ray  A,O,A' which passes through the center of the lens is unchanged because the lens faces are parallel here and the lens is assumed to be thin. This ray is normal to the surface of the lens; (2) a ray A,P,A' parallel to the lens axis is refracted to pass through the focal point F_2 on the opposite side; (3) a ray  A,Q,A' which passes through the focal point F_1 on the side of the object emerges from the lens parallel to the lens axis.

The image is inverted and found on the far side of the focal point away from the lens. The distance from the object to the lens’ nodal plane is BO  and the distance from the image to the lens’ nodal plane is B'O. The height of the real object is H0 and the height of its image is Hi. The object is oriented at a right angle to the lens axis, and so is the image. The geometry of the situation allows us to identify two sets of similar triangles:

1. u = AP = BO  (Distance from the object to the lens’ nodal plane)
 
2. v = A'Q = B'O (Distance from the image to the lens’ nodal plane)

3. f = F1-O = F2-O  (focal length)

4. H0 = AB = PO   (height of the real object)
    
5. Hi = A'B' = QO  (height of its image)
  
From geometry of figure above, triangles ΔABF1 and ΔOQF1 are similar. 

(u-f)/f=H_o/H_i 

Triangles ΔPOF2 and ΔA'B'F2 are also similar 

 f/(v-f)=H_o/H_i    
  
 From the geometry, the thin lens equation may be derived.
 
 1/f=1/v+1/u

  
  
 
  
After we understood lens formula we will make a few edits to lens formula for our project so here are full explain
 
In order to determine the distance from our camera let see the next image


![image](https://user-images.githubusercontent.com/56115477/154845117-cbdf00a6-925d-4265-a3ec-b6da45ac729e.png)


lens formula for a camera is simpler, our camera lens is very small,
so most ray lights coming out from real objec ho ( distance d from camera)
pass through the center of the camera lens(point o) so the ray lights are not refracted and continue straight to the  camera sensor (CMOS).
camera sensor converts the light coming from the lens into electrons, reads each pixel value in the image and so the viewer sees the rael image object  hi.
Rael image object hi with distance f (focal length) from the camera lens ( viewer and the sensor are in the same place).


Definition
1. ho = ho'
2. ak and a'c are rays whit right angle straight angle. 
3. Angle between ok to oa is the same angle between oa' to oc
 we defined angle as θ (This angle is called the field of view, means point of view )

Now from geometry we get:

A. (h_i) / f = tan⁡θ = h_o / d

The only parameter that is unknown is the focal length so:

B. f = (h_i * d) / h_o

The focal length remains constant for our camera, so in order to find its exact value I will double-count the object h_o at different  distances m as shown in the figure.
After finding the exact focal length f, we can know the distance d from our camera to object.
Make few change in our formula we get:
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
d = (f * h_o) / h_i
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉

800 * 600 video display:
![image](https://user-images.githubusercontent.com/56115477/154443892-c8443a8d-e274-4c95-8071-50e83e549bda.png)


Below is what is shown in the figure:
Left figure height of the real object:

 A.     ho≈0.02 [m] 

Figure below  height of real image object (marked in red), to know the height of the real image object in pixels we will multiply the height of real image object by the height of the image which is 800 pixels and we will get:

B.     hi≈800 * 0.12 = 96 [ without units] 

Right figure  distance from real object to our camera:

C.      d≈0.25 [m]

After placing the answers (A, B and C) in formula 2, we get ourfocal length:

f = 1200

After further tests to determine the focal length, it turned out that the exact focal length is:

f = 1170




The result:

![image](https://user-images.githubusercontent.com/56115477/155526624-d8175cd2-1f0f-4698-ae60-1288974285ae.png)

