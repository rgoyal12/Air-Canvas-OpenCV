# Air Canvas Project
### Members- Lochan Gupta, Rajat Goyal, Utkarsh Agrawal
### Mentors- Akshay Gupta, Shivanshu Tyagi
 
 ## Introduction

How wonder it will be if we can draw things by just waving our finger in the air! But this has been made possible by combining digital drawing and smart photo recognition techniques resulting in a hands-free digital drawing canvas called Air Canvas.

We used Python, its libraries like Numpy, Collections, and computer vision techniques of OpenCV to draw anything on the canvas by just capturing the motion of a colored marker with a web camera. We made a basic prototype of Virtual Paint, which enables the user to draw on their system screen by drawing in the air with a bead (we used blue one) which is tracked by the computer webcam. So, to achieve this objective, we used two techniques- Color Detection and Tracking along with the use of masking to keep a track on target bead in real time. 

### Tracking and Paint Canvas Window
![Screenshot (368)](https://user-images.githubusercontent.com/85826885/123541151-99142a80-d760-11eb-80a4-fbf32da14dc3.png)



### Trackbar and Mask Window
![Screenshot (369)](https://user-images.githubusercontent.com/85826885/123541152-9a455780-d760-11eb-978f-08c9be0709f5.png)


 ## Algorithm

1. Start reading the frames using a webcam.
2. Convert the captured BGR frames to HSV color space.
3. Create the Paint window & Canvas frame with putting the respective color palette buttons on it. 
4. Create the trackbar window by adjusting the trackbar values according to the color of the marker bead.
5. Create the mask by using trackbar position values.
6. Processing the mask by using a kernel for smoothening and removing noise with morphological operations(Erosion and dilation).
7. Detect the contours and then find the center coordinates of the largest contour and keep storing them in the deques for successive frames.
8. Finally, draw the points stored in deques on the frames and canvas window with their respective colors.


 ## How We Proceeded
As mentioned above, we first imported all the required libraries, including cv2, NumPy, and deque data structure, which is a part of library collections.

Our whole task was primarily boiled down to three things- namely- color detection, tracking and drawing on the canvas.

 #### **For thecolor detection part**,
- We used OpenCV function “createTrackbar()” used to track colors in HSV space. So, we created six different track bars for upper Hue, upper Saturation, upper Value, lower Hue, lower Saturation, and lower Value where 'Hue' represents the color, 'Saturation'- the amount to which that respective color is mixed with white & 'Value'- the amount to which that respective color is mixed with black (Gray level).
So we set values of parameters in the createTrackbar function parameters by changing the parameter values using trackbar to get the tracking of Blue bead, which we have used as a marker. 

- But the question is why we used HSV instead of working in BGR only.
This is because the HSV model describes colors similar to how the human eye perceives color whereas RGB defines color in terms of combination of primary colors. In RGB, we cannot separate color information from luminance, whereas in HSV, we can separate image luminance from color information. Therefore we preferred HSV on RGB.

#### **For the marker tracking part**
- We made use of data structure deques. On seeing from a webcam's point of view, we would be basically hovering a marker in the 2D plane. So we had to store all the coordinates of our marker (every position that the marker travels) corresponding to each colour. For this purpose, we used the data structure- deques that allows us to push and pop from both ends.
Now along with the marker bead, there comes some impurities with which we got rid of using dilation, and therefore we defined a kernel that would be used in masking.
Then we defined a paintwindow which will be used as a canvas where we used OpenCV functions like cv2.rectangle() and cv2.putText to create boxes for different colors. 


#### **For the drawing on the canvas part** 
- We used cv2.line() function to join between any two points and then we showed the output on the canvas using cv2.imshow() function

### Now comes the actual implementation of our algorithm

- By using cv2.VideoCapture we captured the live frames, and then for processing of each frame, we run a while loop which will end when the user will press the key ‘q’ on the keyboard. 
What we basically did in this while loop is we stored the current live frames and then by creating the mask we detected and stored all the tracked points of marker bead (which it traveled) and then took action on those points according to the respective color we chose.

- Now we flipped the frames (Left-right inversion) captured by the webcam in order to make sure that when we lift our right hand with the marker in it, then in the tracking window(live camera feed), the hand which lifts up will appear to the right in our own’s perspective making it just like a mirror. (This is just for our convenience) 

- In the ‘frame’ variable we saved our current frame and then in ‘hsv’ variable we converted the frame to HSV colour space for separating the luma and intensity that is required in masking for tracking the coordinates of the marker bead.

- Then in arrays ‘Upper_hsv’ and ‘Lower_hsv’ we stored the upper and lower bounds for the color of our marker bead by using the trackbar positions that we have created in start. Here in this case the bound values are for blue color as we have taken our bead color as blue.

- Then in the ‘frame’ window we have created the 5 rectangles for Clear , Blue , Green,Red, Yellow and filled the rectangles with their respective colors. For this we have used the cv2.rectangle() function. After creating and filling them with respective colors we labelled each rectangle  by its respective role by using cv.putText function(). This completes our task of creating the frame window which the user will use to draw in air.

- After this we created a mask window by using the cv2.inRange() function.  In cv2.inRange() function we passed our hsv frame that is stored in ‘hsv’ variable and upper and lower bounds of our marker bead ie.  ‘Upper_hsv’ and ‘Lower_hsv’ arrays.  This function checks the regions of ‘hsv’ frame where the the colour intensity lies between the upper and lower bounds and wherever such regions are found it colour them white and all other regions except these are colored black. 
When doing this we also need to be careful about the noise and smoothing. For this purpose we used some morphological operations like erosion and dilation . Erosion decreases the thickness of edges while dilation increases the thickness of edges. For these operations we needed a kernel which slides down whole the image and changes the value of any pixel  by combining it with different amounts of the neighboring pixels . The kernel used was a matrix of ones of dimension 5 x 5. We used the functions cv2.erode() , cv2.MorphologyEx(), cv2.dilate() for morphological operations. 
So by using the mask we can easily get the regions wherever our marker traveled.

- Now we have a binary image ‘mask’ with regions marked as white tracking the marker’s position. We found all the contours in the mask image using  cv2.findContours() function. Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. cv2.findContours() no longer modifies the source image but returns a modified image as the first of three return parameters ( modified image, the contours and hierarchy). We created a three tuple variable ‘cnts,_ ‘ for storing these three values and in ‘cnts’ variable we have our all contours stored which are just Numpy arrays of (x,y) coordinates of boundary points of the object. 
Then, we declared a variable center which will be later used to store the coordinates of centroid of the largest contour .

- Now in the current frame if we have found some contours that means in this current frame we have moved our maker bead and hence we will have some points and for those points we have to decide what action must be taken. As many contours will be formed but all of them are of not of use . So for resolving this issue we will take the contour of largest length by sorting the contour array in descending order and taking the first element of it. We stored it in ‘cnt’ variable. Around this largest contour we found the coordinates and radius of circle of least radius which encloses it by using cv2.minEnclosingCircle(). We stored the center coordinates in (x,y) and radius in ‘radius’. Now around this contour we drew the above circle with yellow colour by using  cv2.circle() function.

- Now we have to decide for this region of circle that which point in it to select for reference and also it's color. But one problem here is in this region there are infinitely many points and we can’t decide by checking for all the points so we have to take a reference point according to which our decision should be made. The most logical for picking this point is to pick the centroid of the contours. By generating the moments we found the centroid of the contour and stored the centroid coordinates in the ‘ center’ variable.

 ### Maths Behind Moments 
Image moments are a set of statistical metrics that quantify the distribution of pixel locations and intensities. The image moment Mij of order (i,j) for a greyscale image with pixel intensities I(x,y) can be calculated mathematically as
![moment-1](https://user-images.githubusercontent.com/85826885/123540214-93681600-d75b-11eb-8653-a03ec76e790e.jpg)

The row and column indexes are x and y, respectively, and the intensity at that point is I(x,y) (x,y). Let's have a look at how image moments are used to determine basic picture properties.
###### Area:
The area of a binary image corresponds to the zeroth-order moment. Let's discuss how? Using the above formulae, the zeroth-order moment (M00) is given by

![moment_0-1](https://user-images.githubusercontent.com/85826885/123540917-1048bf00-d75f-11eb-84f9-35a1fa4b56af.jpg)

This equates to counting all the non-zero pixels in a binary image, identical to the area. This relates to the sum of pixel intensity values in a greyscale image.
###### Centroid:
The arithmetic mean position of all the points is the centroid. In terms of image moments, the centroid is given by the relation
![moment_cent](https://user-images.githubusercontent.com/85826885/123540927-18086380-d75f-11eb-96fa-cd8e355fbcea.jpg)

This is simple to understand. For instance, for a binary image M10 corresponds to the sum of all non-zero pixels (x-coordinate) and M00 is the total number of non-zero pixels and that is what the centroid is.


### Continuing the Implementation ...
- Now comes the part where we decide what action must be taken for a particular point. As stated above we would use the centroid(i.e. center) as reference for deciding the action.

- **Case 1**
 
    If the y coordinate of center lies where our boxes (Clear,Blue,Green,Red ,Yellow ) are present. It means one of these buttons is clicked . Further five cases are present ,       these five cases represent which button is clicked and it can be easily found by checking the x- coordinate of the center . All cases and their actions are summarised below :-

     If the CLEAR ALL button is pressed we clear all the four deques and set all indexes to 0 indicating that now nothing is present in any of the deques. And also we clear the        paint window means we will color all the paint window white.
     
     If the BLUE button is pressed, we will set our color index to 0 so that the future points will be stored in deque of blue.
     
     If the GREEN button is pressed, we will set our color index to 1 so that the future points will be stored in deque of green.
     
     If the RED button is pressed, we will set our color index to 2 so that the future points will be stored in deque of red.
     
     If the YELLOW button is pressed, we will set our color index to 3 so that the future points will be stored in deque of yellow.

- **Case 2**

    If the y coordinate of the center lies in the drawing region . In this case further 4 cases are possible . They are as follows :

     If color index=0, it means we have to use blue color for drawing . So we will append this center point to the blue deque.
     
     If color index=1, it means we have to use green color for drawing . So we will append this center point to the blue deque.
     
     If color index=2, it means we have to use red color for drawing . So we will append this center point to the blue deque.
     
     If color index=3, it means we have to use yellow for drawing . So we will append this center point to the blue deque.

- It was for the case when we found some contours , but another case which is possible when we have no contours (i.e contour array is empty) . This case indicates that no blue color bead is detected in this frame (here, in the absence of blue bead). So in this case we simply have to do nothing but we can not neglect it as when the bead appears in future frame to another position it does not make a straight line of that color between those two. In this case we will remove the deque of the previous points from each of the four array of deques as all the latest deques must have been processed before reaching this point of time and then we will insert a fresh deque to each of the four arrays so that we can store more points in future.
 
- Finally now we have to show the output  for each frame. We will do this by processing all arrays of deques as every deque of every array may or may not contain some points. So we check for each deque.  If it is not empty we join all the points in the deque by the correct color( it will be known as we know which array this current deque belongs to) in both paint window and frame window by using cv2.line() function. Finally we showed the output using cv2.imshow().

- The loop ends when the user presses the key ‘q’ on the keyboard indicating the end of the process.
  After ending, we released and destroyed all the windows to avoid unnecessary use of space.

