from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import scipy
import scipy.ndimage
import scipy.stats
import pickle
import utils

class Clone(object):
    
    def __init__(self,cloneid,treatment,replicate,rig,datetime,datadir,segdatadir):
        
        self.cloneid = cloneid
        self.treatment = treatment
        self.replicate = replicate
        self.rig = rig
        self.datetime = datetime
         
        delim = "_"
        ext = ".bmp"

        self.filebase = delim.join((cloneid,treatment,replicate,rig,datetime)) + ext
        if os.path.isfile(os.path.join(datadir, "full_" + self.filebase)):
            self.full_filepath = os.path.join(datadir, "full_" + self.filebase)
        
        if os.path.isfile(os.path.join(datadir, "close_" + self.filebase)):
            self.close_filepath = os.path.join(datadir, "close_" + self.filebase)

        if os.path.isfile(os.path.join(datadir, "fullMicro_" + self.filebase)):
            self.micro_filepath = os.path.join(datadir, "fullMicro_" + self.filebase)
        
        if os.path.isfile(os.path.join(segdatadir, "full_" + self.filebase)):
            self.full_seg_filepath = os.path.join(segdatadir, "full_" + self.filebase)

        if os.path.isfile(os.path.join(segdatadir, "close_" + self.filebase)):
            self.close_seg_filepath = os.path.join(segdatadir, "close_" + self.filebase)
        
        self.background_channel = 0
        self.animal_channel = 1
        self.eye_channel = 2
        self.antennae_channel = 3

        self.total_animal_pixels = None
        self.animal_area = None
        
        try:
            self.pixel_to_mm = self.calc_pixel_to_mm(cv2.imread(self.micro_filepath))
        except Exception as e:
            print "Could not calculate pixel image because of the following error: " + str(e)

        self.animal_x_center = None
        self.animal_y_center = None
        self.animal_major = None
        self.animal_minor = None
        self.animal_theta = None
        
        self.eye_x_center = None
        self.eye_y_center = None
        self.eye_major = None
        self.eye_minor = None
        self.eye_theta = None
        
        # these are directional vectors of anatomical direction
        
        self.anterior = None
        self.posterior = None
        self.dorsal = None
        self.ventral = None

        # these are actual points on the animal

        self.eye_dorsal = None
        self.head = None
        self.tail = None
        self.dorsal_point = None

    def crop(self,img):
        
        # this method is for cropping out the scale from micrometer images

        # aperture edges mess up image normalization, so we need to figure out which
        # (if any) corners have aperture edges, as well as how far each of the edges
        # extends (since it is not always symmetric)
        
        w,h = img.shape
        
        corners = []
        docrop = False

        # if there are 5 pixels in a row that are very dark, it is most likely a corner
        if np.sum(img[0, 0:np.int(h/2)] < 50) > 5 and np.sum(img[0:np.int(w/2),0] < 50) > 5:
            docrop = True
            corners.append(["topleft",
                            np.max(np.where(img[0, 0:np.int(h/2)] < 50)),
                            np.max(np.where(img[0:np.int(w/2),0] < 50))])

        if np.sum(img[0, np.int(h/2):] < 50) > 5 and np.sum(img[0:np.int(w/2),h-1] < 50) > 5:
            docrop = True
            corners.append(["topright",
                            np.int(h/2) + np.min(np.where(img[0, np.int(h/2):] < 50)),
                            np.max(np.where(img[0:np.int(w/2),h-1] < 50))])

        if np.sum(img[w-1, np.int(h/2):] < 50) > 5 and np.sum(img[np.int(w/2):,h-1] < 50) > 5:
            docrop = True
            corners.append(["bottomright",
                            np.int(h/2) + np.min(np.where(img[w-1, np.int(h/2):] < 50)),
                            np.int(w/2) + np.min(np.where(img[np.int(w/2):,h-1] < 50))])

        if np.sum(img[w-1,0:np.int(h/2)]<50) >5 and np.sum(img[np.int(w/2):,0] <50) > 5:
            docrop = True
            corners.append(["bottomleft",
                            np.max(np.where(img[w-1,0:np.int(h/2)] < 50)),
                            np.int(w/2) + np.min(np.where(img[np.int(w/2):,0] < 50))])
        
        if len(corners) == 0:
            return img
        else:

            # this method tries to crop the left and righr corners column-wise first
            try:
                leftbound = max([x[1] for x in corners if "left" in x[0]])
            except ValueError:
                leftbound = 0
            
            try:
                rightbound = min([x[1] for x in corners if "right" in x[0]])
            except ValueError:
                rightbound = h-1
            
            if (leftbound > int(h*0.40) or rightbound < int(h*0.6)) or (leftbound == int(h/2)-1 and  rightbound == int(h/2)):

                #if the left and right corners can't be cropped column-wise (e.g. there's a solid border along the bottom)

                if len(corners) == 4:
                    img = cv2.medianBlur(img,5)
                    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                               param1=50,param2=50,minRadius=300)
                    if circles is None:
                        return self.crop(img[int(w/2):,:])
                    else:
                        circle = np.mean(np.array(circles[0]),axis=0)
                        x,y,r = circle
                        return self.crop(img[int(max(y-0.7*r,0)):int(min(y+0.7*r,h)),
                                             int(max(x-0.7*r,0)):int(min(x+0.7*r,w))])
                
                cornernames = [x[0] for x in corners]
                
                if len(corners) == 3:
                    if "topright" not in cornernames:
                        for x in corners:
                            if x[0]=="topleft": leftb = x[1] 
                        for x in corners:
                            if x[0]=="bottomright": lowerb = x[2]
                        return self.crop(img[:lowerb,leftb:])
                    
                    elif "bottomright" not in cornernames:
                        for x in corners:
                            if x[0]=="bottomleft": leftb = x[1]
                        for x in corners:
                            if x[0]=="topright": topb = x[2]
                        return self.crop(img[topb:,leftb:])
                    
                    elif "topleft" not in cornernames:
                        for x in corners:
                            if x[0]=="topright": rightb = x[1]
                        for x in corners:
                            if x[0]=="bottomleft": lowerb = x[2]
                        return self.crop(img[:lowerb,:rightb])
                    
                    elif "bottomleft" not in cornernames:
                        for x in corners:
                            if x[0]=="bottomright": rightb = x[1]
                        for x in corners:
                            if x[0]=="topleft": topb = x[2]
                        return self.crop(img[topb:,:rightb])
                
                elif all(["bottom" in x[0] for x in corners]):
                    threshold = min([x[2] for x in corners])
                    return self.crop(img[0:threshold,:])

                elif all(["top" in x[0] for x in corners]):
                    threshold = max([x[2] for x in corners])
                    return self.crop(img[threshold:,:])

                elif all(["right" in x[0] for x in corners]):
                    threshold = min([x[1] for x in corners])
                    return self.crop(img[:,0:threshold])

                elif all(["left" in x[0] for x in corners]):
                    threshold = max([x[1] for x in corners])
                    return img[:,threshold:]
            else: return self.crop(img[:,leftbound:rightbound])
    
    def dist(self,x,y):

        # returns euclidean distance between two vectors
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x-y)
    
    def find_zero_crossing(self,im,(x1, y1), (x2, y2)):
        
        # finds boundary of binary object (object = 1, background = 0)
        npoints = max(np.abs(y2-y1),np.abs(x2-x1))
        x,y = np.linspace(x1,x2,npoints),np.linspace(y1,y2,npoints)
        zi = scipy.ndimage.map_coordinates(im,np.vstack((x,y)),mode='nearest')

        # this should make the boundary finding more robust to small pockets of mis-classified pixels
        df = pd.DataFrame(zi)
        mva = pd.rolling_mean(zi,8)
        mva = mva[~np.isnan(mva)]

        for i,val in enumerate(mva):
            if val <= 0.05:
                return (x[i],y[i]) 
        return

    def sanitize(self,im):
        try:
            if im.shape[2] == 4:
                return utils.merge_channels(im, self.animal_channel, self.eye_channel)
        except IndexError:
            return im

    def calc_pixel_to_mm(self,im):

        # calculates the pixel to millimeter ratio for a clone given an image of
        # a micrometer associated with clone

        gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cropped = self.crop(gimg)
        
        cl1 = clahe.apply(cropped)
        highcontrast = cl1.copy()

        w,h = highcontrast.shape
        
        edge_threshold = 175
        lines = None

        while edge_threshold > 0 and not np.any(lines):
            edges = cv2.Canny(highcontrast,0,edge_threshold,apertureSize = 3)
            edge_threshold -= 25
            min_line_length = 200

            while min_line_length > 0 and not np.any(lines):
                lines = cv2.HoughLines(edges,1,np.pi/180,200,min_line_length)    
                min_line_length -= 50
        
        if lines is None:
            print "Could not detect ruler"
            return

        measurements = []
        for line in lines[0]:
            rho,theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(gimg,(x1,y1),(x2,y2),(0,0,255),2)
            
            # y=mx+b
            try:
                m = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                continue
            
            b = y2 - m*x2

            x1 = int(0.33*h)
            y1 = int(x1*m + b)
            x2 = int(0.67*h)
            y2 = int(x2*m + b)


            npoints = max(np.abs(y2-y1),np.abs(x2-x1))

            x, y = np.linspace(y1, y2, npoints), np.linspace(x1, x2, npoints)
            # Extract the pixel values along the line
            zi = scipy.ndimage.map_coordinates(highcontrast, np.vstack((x,y)),mode='nearest')
            #mean shift the pixels
            zi = zi-pd.rolling_mean(zi,4)
            df = pd.DataFrame(zi)
            mva = pd.rolling_mean(zi,4)
            mva = mva[~np.isnan(mva)]
            
            #transform to frequency domain
            fourier = np.fft.fft(mva)
            n = fourier.size
            freqs = np.fft.fftfreq(n)
            idx = np.argmax(np.abs(fourier))
            freq = freqs[idx]
            #this is so that really noisy frequencies don't get captured
            try:
                if np.abs(1/freq) < 50:
                    measurements.append(np.abs(1/freq)*40)
            except ZeroDivisionError:
                continue
        return np.mean(measurements)

    def split_channels(self,im):
        
        # splits ilastik segmentation output into 4 channels
        # 1 - background
        # 2 - animal
        # 3 - eye
        # 4 - antennae

        if not np.all(im[:,:,0]==im[:,:,1]) and not np.all(im[:,:,1]==im[:,:,2]):
            print "Can only split segmentation images"
            return im
        
        im = im[:,:,0]
        w,h = im.shape
        channel_ids = np.unique(im)
        nchannels = len(channel_ids)
        
        arrays = list()
        for channel in channel_ids:
            tmp = np.zeros((w,h))
            tmp[np.where(im==channel)] = 1
            arrays.append(tmp)
        
        return np.stack(arrays,axis=2)

    def calculate_area(self,im):
        
        # input:  segmentation image
        # merge animal and eye channels 
        
        try:
            
            animal = self.sanitize(im)

            # count total number of pixels and divide by conversion factor
            self.total_animal_pixels = len(np.flatnonzero(animal))
            self.animal_area = self.total_animal_pixels/(self.pixel_to_mm**2) 
        
        except Exception as e:
            print "Error while calculating area: " + str(e)

    def calculate_length(self):

        try:
            self.animal_length = self.dist(self.head,self.tail)/self.pixel_to_mm
        except Exception as e:
            print e

    def fit_ellipse(self,im,objectType, chi_2):
        
        # fit an ellipse to the animal pixels

        try:
            # input: segmentation image
            # return xcenter,ycenter,major_axis_length,minor_axis_length,theta

            # merge animal and eye channels
            if objectType == "animal":
                ob = self.sanitize(im)
            elif objectType == "eye":
                ob = im[:,:,self.eye_channel]
            else:
                print "Don't know how to fit ellipse to this object type"
                return

            #convert segmentation image to list of points
            points = np.array(np.where(ob))
            n = points.shape[1]
            
            #calculate mean
            mu = np.mean(points,axis=1)
            x_center = mu[0]
            y_center = mu[1]

            #calculate covariance matrix
            z = points.T - mu*np.ones(points.shape).T
            cov = np.dot(z.T,z)/n
            
            #eigenvalues and eigenvectors of covariance matrix correspond
            #to length of major/minor axes of ellipse
            w,v = np.linalg.eig(cov)

            #calculate 90% confidence intervals using eigenvalues to find length of axes
            maj = np.argmax(w)
            minor = np.argmin(w)
            
            major_l = 2*np.sqrt(chi_2*w[maj])
            minor_l = 2*np.sqrt(chi_2*w[minor])

            v = v[maj]
            theta = -np.arctan(v[minor]/v[maj])
            if theta < 0: theta = np.pi/2 - theta

            setattr(self, objectType + "_x_center", x_center)
            setattr(self, objectType + "_y_center", y_center)
            setattr(self, objectType + "_major", major_l)
            setattr(self, objectType + "_minor", minor_l)
            setattr(self, objectType + "_theta", theta)

        except Exception as e:
            print "Error fitting ellipse: " + str(e)
            return

    def fit_animal_ellipse(self,im):

        # this method cleans up any obviously misclassified pixels and re-calculates ellipse

        im = self.sanitize(im)

        self.fit_ellipse(im,"animal",9.21)
        animal = im.copy()
        el = matplotlib.patches.Ellipse((int(self.animal_x_center),int(self.animal_y_center)), int(self.animal_major), int(self.animal_minor),int(self.animal_theta*(180/np.pi)))
        points = list(zip(*(c.flat for c in np.where(animal))))
        
        for i in points:
            if not el.contains_point(i): animal[i] = 0                                               
        
        self.fit_ellipse(animal,"animal",4.6)

    def find_body_landmarks(self,im):

        # this method smooths animal pixels and finds landmarks

        im = self.sanitize(im)

        thresh = cv2.erode(im, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=5)

        self.find_head(im)
        self.find_tail(im)
        self.find_dorsal_point(im)

    def get_anatomical_directions(self):
        
        # finds the vertex points on ellipse fit corresponding to dorsal, ventral, anterior and posterior
        # directions relative to the animal center

        x = self.animal_x_center
        y = self.animal_y_center
        e_x = self.eye_x_center
        e_y = self.eye_y_center
        theta = self.animal_theta
        minor = self.animal_minor
        major = self.animal_major

        major_vertex_1 = (x - 0.5*major*np.sin(theta), y - 0.5*major*np.cos(theta))
        major_vertex_2 = (x + 0.5*major*np.sin(theta), y + 0.5*major*np.cos(theta))

        minor_vertex_1 = (x + 0.5*minor*np.cos(theta), y - 0.5*minor*np.sin(theta))
        minor_vertex_2 = (x - 0.5*minor*np.cos(theta), y + 0.5*minor*np.sin(theta))

        if self.dist((e_x, e_y), major_vertex_1) < self.dist((e_x, e_y), major_vertex_2):
            self.anterior = major_vertex_1
            self.posterior = major_vertex_2
        elif self.dist((e_x, e_y), major_vertex_2) < self.dist((e_x, e_y), major_vertex_1):
            self.anterior = major_vertex_2
            self.posterior = major_vertex_1
 
        if self.dist((e_x, e_y), minor_vertex_1) < self.dist((e_x, e_y), minor_vertex_2):
            self.ventral = minor_vertex_1
            self.dorsal = minor_vertex_2
        elif self.dist((e_x, e_y), minor_vertex_2) < self.dist((e_x, e_y), minor_vertex_1):
            self.ventral = minor_vertex_2
            self.dorsal = minor_vertex_1

    def get_eye_dorsal(self,im):

        # finds dorsal point of the eye
        
        try:
            if im.shape[2] == 4:
                im = im[:,:,self.eye_channel]
        except IndexError:
            pass

        if self.dorsal == None: self.get_anatomical_directions()
        
        if not self.dorsal == None:

            d_y = self.dorsal[1] - self.animal_y_center
            d_x = self.dorsal[0] - self.animal_x_center

            # draw line from eye center with same slope as dorsal axis

            y1 = self.eye_y_center
            x1 = self.eye_x_center
            y2 = self.eye_y_center + d_y
            x2 = self.eye_x_center + d_x
            
            self.eye_dorsal = self.find_zero_crossing(im,(x1,y1),(x2,y2))
    
    def find_head(self, im):

        im = self.sanitize(im)

        if self.anterior is None:
            self.get_anatomical_directions()

        if self.anterior is not None:

            x1 = self.animal_x_center
            y1 = self.animal_y_center

            x2 = 1.5*self.anterior[0] - 0.5*x1
            y2 = 1.5*self.anterior[1] - 0.5*y1

            self.head = self.find_zero_crossing(im, (x1,y1), (x2,y2))

    def find_tail(self, im):
        
        im = self.sanitize(im)

        if self.posterior is None:
            self.get_anatomical_directions()

        if self.posterior is not None:

            x1 = self.animal_x_center
            y1 = self.animal_y_center

            x2 = 1.5*self.posterior[0] - 0.5*x1
            y2 = 1.5*self.posterior[1] - 0.5*y1

            self.tail = self.find_zero_crossing(im, (x1,y1), (x2,y2))
 
    def find_dorsal_point(self, im):
        
        im = self.sanitize(im)

        if self.dorsal is None:
            self.get_anatomical_directions()

        if self.dorsal is not None:

            #x1 = self.animal_x_center
            #y1 = self.animal_y_center
            x_h, y_h = self.head
            x_t, y_t = self.tail
            x1 = np.min(x_h,x_t) + np.abs(0.5*(x_h - x_t))
            y1 = np.min(y_h,y_t) + np.abs(0.5*(y_h - y_t))

            x2 = 1.5*self.dorsal[0] - 0.5*x1
            y2 = 1.5*self.dorsal[1] - 0.5*y1

            self.dorsal_point = self.find_zero_crossing(im, (x1,y1), (x2,y2))   
    
    def slice_pedestal(self,im):
    
        # this method calculates pedestal size (the dumb way)

        im = sanitize(im)

