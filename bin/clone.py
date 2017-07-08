from __future__ import division
import numpy as np
import pandas as pd
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
        self.filebase = delim.join((cloneid,treatment,replicate,rig,datetime))
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
        
        self.anterior = None
        self.posterior = None
        self.dorsal = None
        self.ventral = None

        self.eye_dorsal = None


    def crop(self,img):

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
    
    def dist(x,y):

        # returns euclidean distance between two vectors
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x-y)

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
            if im.shape[2] == 4:
                animal = utils.merge_channels(im,self.animal_channel,self.eye_channel)
                # count total number of pixels and divide by conversion factor
                self.total_animal_pixels = len(np.flatnonzero(animal))
                self.animal_area = self.total_animal_pixels/(self.pixel_to_mm**2) 
        
        except Exception as e:
            print "Error while calculating area: " + str(e)

    def fit_ellipse(self,im,objectType):
        
        try:
            # input: segmentation image
            # return xcenter,ycenter,major_axis_length,minor_axis_length,theta

            # merge animal and eye channels
            if objectType == "animal":
                ob = utils.merge_channels(im,self.animal_channel, self.eye_channel)
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
            
            major_l = 2*np.sqrt(4.6*w[maj])
            minor_l = 2*np.sqrt(4.6*w[minor])

            v = v[maj]
            theta = -np.arctan(v[minor]/v[maj])
            theta_p = np.int(theta/(np.pi/180))

            setattr(self, objectType + "_x_center", int(x_center))
            setattr(self, objectType + "_y_center", int(y_center))
            setattr(self, objectType + "_major", int(major_l))
            setattr(self, objectType + "_minor", int(minor_l))
            setattr(self, objectType + "_theta", int(theta))

        except Exception as e:
            print "Error fitting ellipse: " + str(e)
            return
    
    def get_anatomical_directions(self):
        
        # finds the vertex points on ellipse fit corresponding to dorsal, ventral, anterior and posterior
        # directions relative to the animal center

        major_vertex_1 = (self.animal_y_center - 0.5*self.animal_major*np.sin(animal_theta),
                self.animal_x_center - 0.5*self.animal_major*np.cos(animal_theta))
        major_vertex_2 = (self.animal_y_center + 0.5*self.animal_major*np.sin(animal_theta),
                self.animal_x_center + 0.5*self.animal_major*np.cos(animal_theta))

        minor_vertex_1 = (self.aximal_y_center + 0.5*minor_l*np.sin(theta),
                self.animal_x_center - 0.5*minor_l*np.cos(theta))
        minor_vertex_2 = (self.aximal_y_center - 0.5*minor_l*np.sin(theta),
                self.animal_x_center + 0.5*minor_l*np.cos(theta))

        if dist((self.eye_y_center, self.eye_x_center), major_vertex_1) <
        dist((self.eye_y_center, self.eye_x_center), major_vertex_2):
            self.anterior = major_vertex_1
            self.posterior = major_vertex_2
        elif dist((self.eye_y_center, self.eye_x_center), major_vertex_2) <
        dist((self.eye_y_center, self.eye_x_center), major_vertex_1):
            self.anterior = major_vertex_2
            self.posterior = major_vertex_1
 
        if dist((self.eye_y_center, self.eye_x_center), minor_vertex_1) <
        dist((self.eye_y_center, self.eye_x_center), minor_vertex_2):
            self.ventral = minor_vertex_1
            self.dorsal = minor_vertex_2
 
        elif dist((self.eye_y_center, self.eye_x_center), minor_vertex_2) <
        dist((self.eye_y_center, self.eye_x_center), minor_vertex_1):
            self.ventral = minor_vortex_2
            self.dorsal = minor_vortex_1

    def get_eye_dorsal(self,im):

        # finds dorsal point of the eye

        if im.shape[2] == 4:
            im = im[:,:,self.eye_channel]

        
    def slice_pedestal(self,im):

        #input : segmentation image
        try:
            pass
        except Exception as e:
            print "Error slicing pedestal: " + str(e)
