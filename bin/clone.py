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
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import pickle
import re
import utils
from collections import defaultdict

class Clone(object):
    
    def __init__(self,imtype,barcode,cloneid,treatment,replicate,rig,datetime,induction,datadir,segdatadir):
        
        self.imtype = imtype
        self.cloneid = cloneid
        self.pond = None
        self.id = None
        self.pond, self.id = utils.parsePond(self.cloneid)
        
        if self.cloneid in ["C14","LD33","Chard","D8.4A","D8.6A", "D 8.6A","D8.7A","Cyril"]:
            self.season = "misc"
        elif self.pond == "D8":
            if "AD" in self.cloneid:
                self.season = "spring_2016"
            else: self.season = "spring_2017"
        elif "AW" in self.pond:
            self.season = "spring_2016"
        elif self.pond == "D10":
            if "AD" in self.cloneid:
                self.season = "spring_2016"
            else: self.season = "fall_2016"
        elif self.pond == "DBunk":
            self.season = "spring_2017"
        else:
            self.season = "other"

        if self.cloneid in ["D8_183","D8_191","D8_213","DBunk_90","DBunk_131","DBunk_132"]:
            self.control = True
        else: self.control = False

        self.barcode = barcode
        self.treatment = treatment
        self.replicate = replicate
        self.rig = rig
        self.datetime = datetime
        self.inductiondate = induction
         
        ext = ".bmp"
        delim = "_"
        self.filebase = delim.join((str(barcode),str(cloneid),str(treatment),str(replicate),str(rig),str(datetime))) + ext

        self.convert_treatment()
        
        if os.path.isfile(os.path.join(datadir, imtype + "_" + self.filebase)):
            self.filepath = os.path.join(datadir, imtype + "_" + self.filebase)
        
        if os.path.isfile(os.path.join(datadir, "fullMicro_" + self.filebase)):
            self.micro_filepath = os.path.join(datadir, "fullMicro_" + self.filebase)
        else:
            self.micro_filepath = None

        if os.path.isfile(os.path.join(segdatadir, imtype + "_" + self.filebase)):
            self.seg_filepath = os.path.join(segdatadir, imtype + "_" + self.filebase)
        
        self.background_channel = 0
        self.animal_channel = 1
        self.eye_channel = 2
        self.antennae_channel = 3

        self.total_animal_pixels = None
        self.animal_area = None
        self.total_eye_pixels = None
        self.eye_area = None
        self.animal_length = None
        self.pedestal_size = None
        self.pedestal_height = None
        self.pedestal_score_height = None
        self.pedestal_score_area = None
        self.snake = None
        self.pixel_to_mm = None
        
        #if imtype == "full":
        #    try:
        #        self.pixel_to_mm = self.calc_pixel_to_mm(cv2.imread(self.micro_filepath))
        #    except Exception as e:
        #        print "Could not calculate pixel-to-mm conversion because of the following error: " + str(e)

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
    
        self.analyzed = False

    def convert_treatment(self):
        
        if self.treatment is not None:

            if self.treatment == 'ctrl':
                self.treatment = 0.0
            elif self.treatment == 'juju1':
                self.treatment = 0.1
            elif self.treatment == 'juju2':
                self.treatment = 0.25
            elif (self.treatment == 'juju3') or (self.treatment == 'juju'):
                self.treatment = 0.5
            elif self.treatment == 'juju4':
                self.treatment = 1.0

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
            
            if (leftbound > int(h*0.25) or rightbound < int(h*0.75)) or (leftbound == int(h/2)-1 and  rightbound == int(h/2)):

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
            if im.shape[2] == 3 or im.shape[2] == 4:
                return utils.merge_channels(im, self.animal_channel, self.eye_channel)
        except IndexError:
            return im

    def calc_pixel_to_mm(self,im):

        # calculates the pixel to millimeter ratio for a clone given an image of
        # a micrometer associated with clone

        gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cropped = self.crop(gimg)
        
        w,h = cropped.shape
        cl1 = clahe.apply(cropped)
        highcontrast = cl1.copy()


        edge_threshold = 175
        sum_edges = w*h
        lines = None

        while (edge_threshold > 0 and not np.any(lines)):

            edges = cv2.Canny(highcontrast,0,edge_threshold,apertureSize = 3)
            sum_edges = np.sum(edges)
            edge_threshold -= 25
            min_line_length = 200

            while (min_line_length > 0) and not np.any(lines) and (sum_edges/(255*w*h) < 0.5):
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

    def calculate_area(self, im, part):
        
        # input:  segmentation image
        # merge animal and eye channels 
        
        if part == "animal":
            im = self.sanitize(im)
        elif part == "eye":
            im = im[:, :, self.eye_channel]
        try:     
            # count total number of pixels and divide by conversion factor
            setattr(self, "total_" + part + "_pixels", len(np.flatnonzero(im)))
            setattr(self, part + "_area", getattr(self, "total_" + part + "_pixels")/(self.pixel_to_mm**2))
        
        except Exception as e:
            print "Error while calculating area: " + str(e)

    def calculate_length(self):

        try:
            self.animal_length = self.dist(self.head,self.tail)/self.pixel_to_mm
        except Exception as e:
            print e

    def fit_ellipse(self, im, chi_2):
        
        # fit an ellipse to the animal pixels

        try:
            # input: segmentation image
            # return xcenter,ycenter,major_axis_length,minor_axis_length,theta

            #convert segmentation image to list of points
            points = np.array(np.where(im))
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

            # calculate angle of largest eigenvector towards the x-axis to get theta relative to x-axis
            v = v[minor]
            theta = np.arctan(v[1]/v[0])
            
            return x_center, y_center, major_l, minor_l, theta

        except Exception as e:
            print "Error fitting ellipse: " + str(e)
            return

    def fit_animal_ellipse(self,im):

        # this method cleans up any obviously misclassified pixels and re-calculates ellipse

        im = self.sanitize(im)

        self.animal_x_center, self.animal_y_center, self.animal_major, self.animal_minor, self.animal_theta = self.fit_ellipse(im,9.21)

        animal = im.copy()
        el = matplotlib.patches.Ellipse((int(self.animal_x_center),int(self.animal_y_center)), int(self.animal_major), int(self.animal_minor),int(self.animal_theta*(180/np.pi)))
        points = list(zip(*(c.flat for c in np.where(animal))))
        
        for i in points:
            if not el.contains_point(i): animal[i] = 0                                               
        
        self.animal_x_center, self.animal_y_center, self.animal_major, self.animal_minor, self.animal_theta = self.fit_ellipse(animal,9.21)

    def fit_eye_ellipse(self,im):

        if im.shape[2] == 4:
            im = im[:, :, self.eye_channel]

        self.eye_x_center, self.eye_y_center, self.eye_major, self.eye_minor, self.eye_theta = self.fit_ellipse(im, 4.6)

    def find_body_landmarks(self,im,segim):
        if len(im.shape) > 1:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # before merging channels, find eye landmarks:
        self.find_eye_vertex(segim, "dorsal")
        self.find_eye_vertex(segim, "anterior")
        self.head = self.eye_anterior
        # this method smooths animal pixels and finds landmarks

        self.find_tail(im)
        #self.find_head(im,segim)
        #except TypeError as e:
        #    print "Error in finding body landmarks"
        #    pass
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
        
        if (self.dorsal[0] < 0) or (self.dorsal[0] > 768) or (self.dorsal[1] < 0) or (self.dorsal[1] > 1024):
            tmp = self.dorsal
            self.dorsal = self.ventral
            self.ventral = tmp

    def find_eye_vertex(self, im, vertex):

        # finds dorsal point of the eye
        
        if vertex not in ["dorsal", "ventral", "anterior", "posterior"]:
            raise( "Choose different (and real) direction")

        try:
            if im.shape[2] == 4 or im.shape[2] == 3:
                im = im[:,:,self.eye_channel]

        except IndexError:
            pass

        body_vertex = getattr(self, vertex)

        if body_vertex == None:
            self.get_anatomical_directions()
        else:

            d_y = body_vertex[1] - self.animal_y_center
            d_x = body_vertex[0] - self.animal_x_center

            # draw line from eye center with same slope as dorsal axis

            y1 = self.eye_y_center
            x1 = self.eye_x_center
            y2 = self.eye_y_center + d_y
            x2 = self.eye_x_center + d_x
            
            setattr(self, "eye_" + vertex,  self.find_zero_crossing(im,(x1,y1),(x2,y2)) )
    
    def find_head(self, im, segim):

        if self.tail is None:
            self.find_tail(im)
        
        if self.eye_dorsal is None:
            self.find_eye_dorsal(segim)

        if (self.tail is not None) and (self.eye_dorsal is not None):

            x1 = self.eye_dorsal[0]
            y1 = self.eye_dorsal[1]
           
            idx_x, idx_y = self.gradient_mask(im, "head",0.35)
            
            mask = im.copy()
            mask[idx_x, idx_y] = 0

            x1 = self.eye_dorsal[0]
            y1 = self.eye_dorsal[1]

            m = (self.tail[1] - y1)/(self.tail[0] - x1)
            b = y1 - x1*m

            x2 = 1.25*self.anterior[0] - 0.25*self.animal_x_center
            y2 = m*x1 + b

            npoints = max(np.abs(y2 - y1), np.abs(x2 - x1))*2
            x,y = np.linspace(x1,x2,npoints), np.linspace(y1,y2,npoints)

            z = scipy.ndimage.map_coordinates(mask, np.vstack((x,y)), mode='nearest')
            df = pd.DataFrame(z)

            hx = list()
            hy = list()

            found = False

            for i in df.iterrows():
                if i[1][0] == 0:
                    hx.append(x[i[0]])
                    hy.append(y[i[0]])

                    found = True

            if not found:
                hx, hy = x1, y1
            else:
                hx = np.mean(hx)
                hy = np.mean(hy)

            self.head = hx, hy

    def find_tail(self, im):
        
        # uses ellipse fit to find tail landmark
        #
        # input: grayscale raw image

        idx_x, idx_y = self.gradient_mask(im, "tail")
        
        targetx = self.posterior[0] + self.dorsal[0] - self.animal_x_center
        targety = self.posterior[1] + self.dorsal[1] - self.animal_y_center

        tail_idx = np.argmin(np.sqrt((idx_x - targetx)**2 + (idx_y - targety)**2))
        self.tail = [idx_x[tail_idx], idx_y[tail_idx]]

    def gradient_mask(self, im, part, threshold=0.3):

        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if part == "tail":
            x, y = self.posterior
        elif part == "head":
            x, y = self.anterior

        w, h = im.shape

        t = 100
        bb = np.zeros((4,2))
        bb[0, :] = [np.max([x-t, 0]), np.max([y-t, 0])]
        bb[1, :] = [np.min([x+t, w]), np.max([y-t, 0])]
        bb[2, :] = [np.max([x-t, 0]), np.min([y+t, h])]
        bb[3, :] = [np.min([x+t, w]), np.min([y+t, h])]

        cropped = im[int(bb[0,0]):int(bb[3,0]), int(bb[0,1]):int(bb[3,1])]

        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        hc = clahe.apply(cropped)

        blur = gaussian(hc, 1.5)
        dx, dy = np.gradient(blur)
        
        posterior = [self.animal_x_center - self.posterior[0], self.animal_y_center - self.posterior[1]]
        dorsal = [self.animal_x_center - self.dorsal[0], self.animal_y_center - self.dorsal[1]]
        ventral = [self.animal_x_center - self.ventral[0], self.animal_y_center - self.ventral[1]]
        anterior = [self.animal_x_center - self.anterior[0], self.animal_y_center - self.anterior[1]]
        
        if part == "tail":
            post_dot = posterior[0]*dx + posterior[1]*dy
            idx_x, idx_y = np.where(utils.norm(post_dot) < threshold)
        if part == "head":
            ant_dot = anterior[0]*dx + anterior[1]*dy
            idx_x, idx_y = np.where(utils.norm(ant_dot) < threshold)

        idx_x += int(bb[0,0])
        idx_y += int(bb[0,1])

        return idx_x, idx_y

    def initialize_snake(self):

        head = self.head
        tail = self.tail
        mp = (head[0] + tail[0])/2, (head[1] + tail[1])/2
        dp = (self.dorsal[0] + mp[0])/2, (self.dorsal[1] + mp[1])/2
        dvec = self.animal_x_center - self.dorsal[0], self.animal_y_center - self.dorsal[1]
        diameter = self.dist(head, dp)
        cx, cy = (head[0] + dp[0])/2, (head[1] + dp[1])/2

        if (self.animal_x_center - self.anterior[0] < 0):
            theta = np.arctan2(dp[0] - cx, dp[1] - cy)
            s = np.linspace(theta, theta - np.sign(dvec[1])*np.pi, 400)

        elif (self.animal_x_center - self.anterior[0] > 0):
            theta = np.arctan2(head[0] - cx, head[1] - cy)
            s = np.linspace(theta, theta - np.sign(dvec[1])*np.pi, 400)

        x = cy + int(diameter/2)*np.cos(s)
        y = cx + int(diameter/2)*np.sin(s)

        self.pedestal_snake =  np.array([x, y]).T
