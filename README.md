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
