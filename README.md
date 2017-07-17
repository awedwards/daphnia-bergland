# daphnia
Daphnia image analysis

Automated image analysis for extracting features from whole images of <i>Daphnia magna</i>.

Here is a description of some of the main features of <b>daphnia</b>:

## Extracting pixel-to-millimeter conversion

In our experimental pipeline, an image of a micrometer is captured before imaging each <i>Daphnia</i> clone.

![alt text](https://github.com/awedwards/daphnia/blob/master/media/raw_micro.png)

The aperture edges cause a lot of issues, so we want to crop the scale:

![alt text](https://github.com/awedwards/daphnia/blob/master/media/cropped_micro.png)

Next, we use a Hough line detector to find the through-line in the scale:

![alt text](https://github.com/awedwards/daphnia/blob/master/media/cropped_micro_line_detection.png)

Finally, we extract the intensity value along the line and perform a Fourier transform to calculate the pixel frequency. This gives us the number of pixels between each scale mark.

![alt text](https://github.com/awedwards/daphnia/blob/master/media/pixeltomm.png)


## Daphnia analysis

Once we have the pixel-to-millimeter conversion factor, the analysis we do for the animal will mean something!

To do most of the analyses, we will need a segmented image of the <i>Daphnia</i>. I used [ilastik](https://github.com/ilastik). The input to most methods will be the segmentation image, which looks something like this (with all channels merged):

![alt text](https://github.com/awedwards/daphnia/blob/master/media/segmentation.png)

We want to fit an ellipse to the animal to find the anatomical directions. The fit_animal_ellipse method does this iteratively, so that we can remove any pixels that have obviously been mis-classified. Then we can calculate features like size of the animal:

![alt text](https://github.com/awedwards/daphnia/blob/master/media/cleaned_animal_area.png)

OpenCV provides a way to erode any small blobs in an image. I use this to further clean the segmentation:

![alt text](https://github.com/awedwards/daphnia/blob/master/media/cleaned_animal_threshold.png)

Now we can use the eye as a landmark to find the anatomical axes of the animal

![alt text](https://github.com/awedwards/daphnia/blob/master/media/anatomical.png)

We can use these directions to find landmarks on the actual animal. First, the "tail" of the animal is found by traversing along the posterior of the animal until a body boundary is found (this could be updated to include information from the actual tail, which I have labeled as "antennae" in the segmentation step). Next, the head of the animal is found by drawing a line between the tail and the dorsal point of the eye, and continuing until a body boundary is found.

![alt text](https://github.com/awedwards/daphnia/blob/master/media/landmarks.png)

Length of the animal can be calculated from the head/tail landmarks:

![alt text](https://github.com/awedwards/daphnia/blob/master/media/length.png)

To do (in order of priority):
- Calculate pedestal size by drawing a line from head point to dorsal point (see two images up), counting pixels, and then normalizing by body size
- Plotting this stuff! Because there are so many clones/replicates, would love to get some ideas on how to group things. For instance, for length/whole size, I guess we don't care about treatment? So we could group those together if we just want to see a size progression.
- Work on close-up images to see if we can count neck teeth
- Think about different ways of quantifying pedestal shape
- Extracting other features that might be interesting
