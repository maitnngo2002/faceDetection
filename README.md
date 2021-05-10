# This is the repo for my face detection model

**NOTE**: Face detection is different from face recognition (face verification)
      
 Face detection is to **detect faces** in a(n) image/video while face recognition is to **match a human face** in a(n) image/video **against a database of faces**


### Techniques that I used

I implement the face detection model by two ways ( **Cascade Classifier** and **Histogram of oriented gradients ( HOG)**

For the first way, check the file name *face_detection*.
For the second way, check the file name *face_detection_using_HOG*

## Haar Cascade Classifier

To detect face in an image -> use **Haar Cascades Classifier** in which a cascade function is trained from lots of image (both positive and negative) and is used to detect objects/ faces in other images

**Haar Cascades Classifier** works as a convolutional kernel which extracts features from images using some sort of filters (called *Haar features). 

It seems inefficient for the training process if the amount of features becomes large. So here is when **Integral Image** comes into play. 

It is an algorithm for quick and efficient computation of the sum of values in a rectangle subset of a pixel grid.
It works by **passing over the image** which means **every new pixel is the sum of all pixels above and to the left of it including it**

On the other parts, we need to know **which features are relevant** and which ones to get rid of -> we use **Adaboost** - a boosting technique to select the best features and train the classifier to use them! For example, **a vertical edge can detect noses not lips**


Luckily, OpenCV has all above algorithms pretrained for us!


## Histogram of oriented gradients (HOG)

To implement this technique we must install **Dlib** library -->> Follow this [Guide](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f)

The idea is to extract features into a vector which will then be fed into a classification algorithm like Support Vector Machine (SVM) to access whether a face is present in a region or not ( **person/ non-person classification**)

Firstly, we must ensure all images are of the same size by cropping and rescaling them to ratio 1 : 2

Next, compute gradient images by applying **Sobel filter**

Then, we divide the image into 8x8 cells and **compute HOG for each cell**. To estimate the direction of a gradient inside a region, we build a histogram among 64 values of gradient directions and their magnitude (another 64 values) inside each region

The final step is **block normalization** - dividing each value of the HOG of size 8x8 by the L2-norm of the HOG of the 16x16 block that contains it, which is in fact a simple vector of length 9 x 4 = 36.
And all 36 x 1 vectors are concatenated into a large vector
