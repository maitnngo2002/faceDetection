### This is the repo for my face detection model

**NOTE**: Face detection is different from face recognition (face verification)
      
          Face detection is to **detect faces** in a(n) image/video while face recognition is to **match a human face** in a(n) image/video **against a database of faces**

# Install necessary libraries

To start, you must **install Dlib and OpenCV**

With OpenCV, you just need to type into your terminal `pip install opencv-python`

To install Dlib, follow this [Guide](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f)

# Techniques that I used

To detect face in an image -> use **Haar Cascades Classifier** in which a cascade function is trained from lots of image (both positive and negative) and is used to detect objects/ faces in other images

**Haar Cascades Classifier** works as a convolutional kernel which extracts features from images using some sort of filters (called *Haar features). 

It seems inefficient for the training process if the amount of features becomes large. So here is when **Integral Image** comes into play. 

It is an algorithm for quick and efficient computation of the sum of values in a rectangle subset of a pixel grid.
It works by **passing over the image** which means **every new pixel is the sum of all pixels above and to the left of it including it**

On the other parts, we need to know **which features are relevant** and which ones to get rid of -> we use **Adaboost** - a boosting technique to select the best features and train the classifier to use them! For example, **a vertical edge can detect noses not lips**


Luckily, OpenCV has all above algorithms pretrained for us!
