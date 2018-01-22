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
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import pickle
import re
import utils
from collections import defaultdict

class Clone(object):
    
    def __init__(self,filebase,imtype,barcode,cloneid,treatment,replicate,rig,datetime,induction,datadir):
        
        self.imtype = imtype
        self.cloneid = cloneid
        self.pond = None
        self.id = None
        self.pond, self.id = utils.parsePond(self.cloneid)
        self.sampling = None

        if self.cloneid in ["C14","LD33","Chard","D8_4A","D8_6A", "D8_6A","D8_7A","Cyril"]:
            self.season = "misc"
        elif self.pond == "AD8":
            self.season = "spring_2016"
        elif self.pond == "D8":
            self.season = "spring_2017"
        elif "AW" in self.pond:
            self.season = "spring_2016"
        elif self.pond == "AD10":
            self.season = "spring_2016"
        elif self.pond == "D10":
            self.season = "fall_2016"
        elif self.pond == "DBunk":
            self.season = "spring_2017"
        else:
            self.season = "other"
        
        if self.season == "spring_2017":
            if (self.pond == "D8") or (self.pond == "DBunk"):
                if int(self.id) < 500:
                    self.sampling = "first_sampling"
                else: self.sampling = "second_sampling"

        if self.cloneid in ["D8_183","D8_191","D8_213","DBunk_90","DBunk_131","DBunk_132"]:
            self.control = True
        else: self.control = False

        self.barcode = barcode
        self.treatment = treatment
        self.replicate = replicate
        self.rig = rig
        self.datetime = datetime
        self.inductiondate = induction
         
        self.filebase = filebase

        self.convert_treatment()
        
        if os.path.isfile(os.path.join(datadir, imtype + "_" + self.filebase)):
            self.filepath = os.path.join(datadir, imtype + "_" + self.filebase)
        
        if os.path.isfile(os.path.join(datadir, "fullMicro_" + self.filebase)):
            self.micro_filepath = os.path.join(datadir, "fullMicro_" + self.filebase)
        else:
            self.micro_filepath = None

        self.total_animal_pixels = None
        self.animal_area = None
        self.total_eye_pixels = None
        self.eye_area = None
        self.animal_length_pixels = None
        self.animal_length = None
        self.pedestal_size = None
        self.pedestal_max_height = None
        self.pedestal_area = None
        self.pedestal_theta = None
        self.snake = None
        self.pixel_to_mm = None
        

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

        # these are directional vectors of anatomical direction starting at origin
        
        self.anterior = None
        self.posterior = None
        self.dorsal = None
        self.ventral = None
        
        # these are directional vectors of anatomical direction starting at animal center
        self.ant_vec = None
        self.pos_vec = None
        self.dor_vec = None
        self.ven_vec = None

        # endpoints for masking antenna
        self.ventral_mask_endpoints = None
        self.dorsal_mask_endpoints = None
        self.anterior_mask_endpoints = None
        self.posterior_mask_endpoints = None

        # these are actual points on the animal

        self.eye_dorsal = None
        self.head = None
        self.tail = None
        self.tail_tip = None
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

    def find_eye(self, im, sigma=0.5):

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, sigma), dtype=np.uint8), 0, 50)/255

        # initialize eye center
        eye_im = np.where((im < np.percentile(im, 0.025)))
        ex, ey = np.median(eye_im, axis=1)

        to_check = [(int(ex), int(ey))]
        checked = []
        eye = []

        count = 0

        while len(to_check)>0:
            pt = to_check[0]
            if (edges[pt[0]-1, pt[1]] == 0) and (edges[pt[0]+1, pt[1]] == 0) and (edges[pt[0], pt[1]-1] == 0) and (edges[pt[0], pt[1]+1] == 0):
                count +=1
                eye.append((pt[0], pt[1]))
                if ((pt[0]-1, pt[1]) not in checked) and ((pt[0]-1, pt[1]) not in to_check):
                        to_check.append((pt[0]-1, pt[1]))
                if ((pt[0]+1, pt[1]) not in checked) and ((pt[0]+1, pt[1]) not in to_check):
                        to_check.append((pt[0]+1, pt[1]))
                if ((pt[0], pt[1]-1) not in checked) and ((pt[0], pt[1]-1) not in to_check):
                        to_check.append((pt[0], pt[1]-1))
                if ((pt[0], pt[1]+1) not in checked) and ((pt[0], pt[1]+1) not in to_check):
                        to_check.append((pt[0], pt[1]+1))
            
            checked.append(to_check.pop(0))
        

        self.eye_pts = np.array(eye)
        try:
            self.eye_x_center, self.eye_y_center = np.mean(np.array(eye), axis=0)
            self.total_eye_pixels = count
        except (TypeError, IndexError):
            self.find_eye(im, sigma=sigma+0.25)
    
    def get_eye_area(self):

        self.eye_area = self.total_eye_pixels/np.power(self.pixel_to_mm, 2)

    def count_animal_pixels(self, im, sigma=1, canny_thresholds=[0,50]):
        
        cx, cy = self.animal_x_center, self.animal_y_center

        hx1, hy1 = self.ventral_mask_endpoints[0]
        vx, vy = self.ventral_mask_endpoints[1]
        
        hx2, hy2 = self.dorsal_mask_endpoints[0]
        dx, dy = self.dorsal_mask_endpoints[1]

        topx1, topy1 = self.anterior_mask_endpoints[0]
        topx2, topy2 = self.anterior_mask_endpoints[1]

        r = 2*self.dist((cx, cy), self.anterior)

        s = np.linspace(0, 2*np.pi, 100)
        x = cx + int(r)*np.sin(s)
        y = cy + int(r)*np.cos(s)

        pts = []

        grad_x, grad_y = np.gradient(gaussian(im, sigma=sigma))

        for i in xrange(len(s)):
            
            dot = grad_x*(cx - x[i]) + grad_y*(cy - y[i])

            p1 = (y[i], x[i])
            p2 = (cy, cx)

            if self.intersect((p1[1], p1[0], cx, cy), (hx1, hy1, vx, vy)):
                res = self.intersection((p1[1], p1[0], cx, cy), ((hx1, hy1, vx, vy)))
                p1 = (res[1], res[0])

            if self.intersect((p1[1], p1[0], cx, cy), (hx2, hy2, dx, dy) ):
                res = self.intersection((p1[1], p1[0], cx, cy), (hx2, hy2, dx, dy))
                p1 = (res[1], res[0])

            if self.intersect((p1[1], p1[0], cx, cy), (topx1, topy1, topx2, topy2) ):
                res = self.intersection((p1[1], p1[0], cx, cy), (topx1, topy1, topx2, topy2))
                p1 = (res[1], res[0])

            edge_pt = self.find_edge(dot, p1, p2)

            if edge_pt is not None:
                pts.append((edge_pt[1], edge_pt[0]))

        pts = np.array(pts)
        cc = [[]]
        idx = 0
        connected = False

        for i in xrange(1, pts.shape[0] - 1):

            if (self.dist(pts[i, :], pts[i-1,:]) < 15) or (self.dist(pts[i+1, :], pts[i,:]) < 15):

                try:
                    cc[idx].append(pts[i,:])
                    connected = True

                except IndexError:
                    cc.append([])
                    cc[idx].append(pts[i,:])
                    connected = True
            else:
                try:
                    if len(cc[idx]) < 4:
                        cc.pop(idx)
                        idx -= 1
                except IndexError:
                    pass

                else:
                    if connected:
                        idx += 1
                        connected = False

        self.whole_animal_points = cc
        cc = np.vstack(cc)
        self.total_animal_pixels = self.area(cc[:,1], cc[:,0])

    def get_animal_area(self):

        self.animal_area = self.total_animal_pixels/np.power(self.pixel_to_mm, 2)

    def get_animal_length(self):

        self.animal_length_pixels = self.dist(self.eye_dorsal, self.tail)
        self.animal_length = self.animal_length_pixels/self.pixel_to_mm

    def mask_antenna(self, im, sigma=1.5, canny_thresholds=[0,50], cc_threhsold=125, a = 0.7, b=20, c=2):
        ex, ey = self.eye_x_center, self.eye_y_center

        high_contrast_im = self.high_contrast(im)

        edge_image = cv2.Canny(np.array(255*gaussian(high_contrast_im, sigma), dtype=np.uint8), canny_thresholds[0], canny_thresholds[1])/255
        edge_labels = measure.label(edge_image, background = 0)

        edge_copy = edge_image.copy()
        
        labels = np.ndarray.flatten(np.argwhere(np.bincount(np.ndarray.flatten(edge_labels[np.nonzero(edge_labels)])) > cc_threhsold))
        big_cc = np.isin(edge_labels, labels) 
        big_cc_x = np.where(big_cc)[0]
        big_cc_y = np.where(big_cc)[1]

        idx = np.argmax(np.linalg.norm(np.vstack(np.where(big_cc)).T - np.array((self.eye_x_center, self.eye_y_center)), axis=1))
        tx = big_cc_x[idx]
        ty = big_cc_y[idx]

        self.tail_tip = (tx, ty)

        cx, cy = (tx + ex)/2, (ty + ey)/2

        hx1, hy1 = 1.2*(ex - cx) + cx, 1.2*(ey - cy) + cy

        vd1 = cx + a*(hy1 - cy), cy + a*(cx - hx1)
        vd2 = cx - a*(hy1 - cy), cy - a*(cx - hx1)

        hx2, hy2 = 1.125*(ex - cx) + cx, 1.125*(ey - cy) + cy
        top1 = hx2 + b*(ey - hy2), hy2 + b*(hx2 - ex)
        top2 = hx2 - b*(ey - hy2), hy2 - b*(hx2 - ex)

        self.tail = 0.4*cx + 0.6*self.tail_tip[0], 0.4*cy + 0.6*self.tail_tip[1]
        bot1 = self.tail[0] + c*(self.tail_tip[1] - self.tail[1]), self.tail[1] + c*(self.tail_tip[0] - self.tail[0])
        bot2 = self.tail[0] - c*(self.tail_tip[1] - self.tail[1]), self.tail[1] - c*(self.tail_tip[0] - self.tail[0])
       
        edges_x = np.where(edge_image)[0]
        edges_y = np.where(edge_image)[1]

        #TO DO: Vectorize
        mask_x1 = []
        mask_y1 = []
        mask_x2 = []
        mask_y2 = []
        top_mask_x = []
        top_mask_y = []

        for i in xrange(len(edges_x)):
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [hx1, hy1, vd1[0], vd1[1]]):
                mask_x1.append(edges_x[i])
                mask_y1.append(edges_y[i])
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [hx1, hy1, vd2[0], vd2[1]]):
                mask_x2.append(edges_x[i])
                mask_y2.append(edges_y[i])
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [top1[0], top1[1], top2[0], top2[1]]):
                top_mask_x.append(edges_x[i])
                top_mask_y.append(edges_y[i])

        edge_copy[[mask_x1, mask_y1]] = 0
        edge_copy[[mask_x2, mask_y2]] = 0
        edge_copy[[top_mask_x, top_mask_y]] = 0

        self.get_anatomical_directions(edge_copy)

        if self.dist( self.ventral, vd1 ) < self.dist( self.ventral, vd2 ):
            self.ventral_mask_endpoints = ((hx1, hy1), vd1)
        else:
            self.ventral_mask_endpoints = ((hx1, hy1), vd2)
        
        hx, hy = self.ventral_mask_endpoints[0]
        vx, vy = self.ventral_mask_endpoints[1]

        m = (hy - vy)/(hx - vx)
        b = hy - m*hx

        x2 = (cx + m*(cy-b))/(1 + m**2)
        y2 = (m*cx + (m**2)*cy + b)/(1 + m**2)

        shift = np.array( (x2 - cx, y2 - cy) )

        self.dorsal_mask_endpoints = ((hx - 1.4*shift[0], hy - 1.4*shift[1]), self.tail_tip)
        self.ventral_mask_endpoints = ((self.ventral_mask_endpoints[0][0] + 0.05*shift[0],
            self.ventral_mask_endpoints[0][1] + 0.05*shift[1]),
            (self.ventral_mask_endpoints[1][0] + 0.05*shift[0],
                self.ventral_mask_endpoints[1][1] + 0.05*shift[1]))
        self.anterior_mask_endpoints = (top1, top2)
        self.posterior_mask_endpoints = (bot1, bot2)
        
    def get_anatomical_directions(self, im, sigma=3, flag="animal"):

        x, y, major, minor, theta = self.fit_ellipse(im, sigma)
        self.animal_x_center, self.animal_y_center, self.animal_major, self.animal_minor, self.animal_theta = x, y, major, minor, theta
        
        major_vertex_1 = (x - 0.5*major*np.sin(theta), y - 0.5*major*np.cos(theta))
        major_vertex_2 = (x + 0.5*major*np.sin(theta), y + 0.5*major*np.cos(theta))

        minor_vertex_1 = (x + 0.5*minor*np.cos(theta), y - 0.5*minor*np.sin(theta))
        minor_vertex_2 = (x - 0.5*minor*np.cos(theta), y + 0.5*minor*np.sin(theta))
        

        if self.dist( major_vertex_1, (self.eye_x_center, self.eye_y_center)) < self.dist(major_vertex_2, (self.eye_x_center, self.eye_y_center)):
            self.anterior = major_vertex_1
            self.posterior = major_vertex_2
        else:
            self.anterior = major_vertex_2
            self.posterior = major_vertex_1

        if self.dist( minor_vertex_1, self.tail_tip ) < self.dist(minor_vertex_2, self.tail_tip):
            self.dorsal = minor_vertex_1
            self.ventral = minor_vertex_2
        else:
            self.dorsal = minor_vertex_2
            self.ventral = minor_vertex_1

    def get_orientation_vectors(self):

        self.pos_vec = [self.animal_x_center - self.posterior[0], self.animal_y_center - self.posterior[1]]
        self.dor_vec = [self.animal_x_center - self.dorsal[0], self.animal_y_center - self.dorsal[1]]
        self.ven_vec = [self.animal_x_center - self.ventral[0], self.animal_y_center - self.ventral[1]]
        self.ant_vec = [self.animal_x_center - self.anterior[0], self.animal_y_center - self.anterior[1]]
    
    def get_eye_vector(self, vec):

        # finds dorsal point of the eye
        
        if vec not in ["dorsal", "ventral", "anterior", "posterior"]:
            raise( "Direction not found")

        eye = self.eye_pts

        body_vector = getattr(self, vec)

        if body_vector == None:
            self.get_anatomical_directions()

        d_x = body_vector[0] - self.animal_x_center
        d_y = body_vector[1] - self.animal_y_center
        
        # draw line from eye center with same slope as body vector

        x = self.eye_x_center + d_x 
        y = self.eye_y_center + d_y
         
        setattr(self, "eye_" + vec, tuple(eye[np.argmin(np.linalg.norm(eye -  np.array((x, y)))), :]) )
    
    def find_head(self, im):

        if self.tail is None:
            self.find_tail()
        
        if self.eye_dorsal is None:
            self.get_eye_vector(im, "dorsal")

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

            hx = []
            hy = []

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

    def find_dorsal(self, im):

        dot = self.gradient(im, "dorsal")

        p1 = ((self.head[0] + self.tail[0])/2, (self.head[1] + self.tail[1])/2)
        m = -1/((self.head[1] - self.tail[1])/(self.head[0] - self.tail[0]))
        b = p1[1] - m*p1[0]
        p2 = (self.dorsal[1]-b)/m, self.dorsal[1]
        self.dorsal_point = self.find_edge(dot, p2, p1)

    def find_tail(self):
        
        # get ventral/posterior-most connected components from the whole animal fitting,
        # then get the point in that set closest to the tail tip
        
        self.get_orientation_vectors()
        ven_pos_vec = np.array(self.pos_vec) + np.array(clone.ven_vec)
        cx, cy = self.animal_x_center, self.animal_y_center

        mean_dist = []

        for c in self.whole_animal_points:

            m = np.mean( c, axis = 0 )
            mean_dist.append( self.dist( m, ( cx - ven_pos_vec[0], cy - ven_pos_vec[1]) ) )

        ven_pos_points = np.vstack( self.whole_animal_points[np.argmin( mean_dist )] )
        
        idx = np.argmin(np.linalg.norm(ven_pos_points - np.array((self.tail_tip[0], self.tail_tip[1])), axis=1))
        
        self.tail = (ven_pos_points[idx, 0], ven_pos_points[idx, 1])

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
        
        if part == "tail":
            post_dot = self.pos_vec[0]*dx + self.pos_vec[1]*dy
            idx_x, idx_y = np.where(self.norm(post_dot) < threshold)
        if part == "head":
            ant_dot = self.ant_vec[0]*dx + self.ant_vec[1]*dy
            idx_x, idx_y = np.where(self.norm(ant_dot) < threshold)

        idx_x += int(bb[0,0])
        idx_y += int(bb[0,1])

        return idx_x, idx_y

    def initialize_snake(self, im):
        
        ex, ey = self.eye_x_center, self.eye_y_center
        tx, ty = self.tail[0], self.tail[1]

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, 1.25), dtype=np.uint8), 0, 50)/255

        m = (ty - ey)/(tx - ex)
        b = ey - m*ex

        d = clone.dist((ex, ey), (self.dorsal_mask_endpoints[0][0], self.dorsal_mask_endpoints[0][1]))
        
        x1 = ex + np.sqrt((d**2)/(1 + 1/(m**2)))
        y1 = ey - (1/m)*(x1 - ex)

        x2 = ex - np.sqrt((d**2)/(1 + 1/(m**2)))
        y2 = ey - (1/m)*(x2 - ex)
        
        if self.dist((x1, y1), (self.dorsal[0], self.dorsal[1])) < self.dist((x2, y2), (self.dorsal[0], self.dorsal[1])):
            p1 = x1, y1
        else:
            p1 = x2, y2
        
        p1 = self.find_edge2(edges, p1, (ex, ey))
        
        mp = 0.67*ex + 0.33*tx, 0.67*ey + 0.33*ty

        x1 = mp[0] + np.sqrt((d**2)/(1 + 1/(m**2)))
        y1 = mp[1] - (1/m)*(x1 - mp[0])

        x2 = mp[0] - np.sqrt((d**2)/(1 + 1/(m**2)))
        y2 = mp[1] - (1/m)*(x2 - mp[0])
        
        if self.dist((x1, y1), (self.dorsal[0], self.dorsal[1])) < self.dist((x2, y2), (self.dorsal[0], self.dorsal[1])):
            p2 = x1, y1
        else:
            p2 = x2, y2
        
        p2 = self.find_edge2(edges, p2, mp)

        diameter = self.dist(p1, p2)
        cx, cy = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2

        if (self.animal_x_center - self.anterior[0] < 0):
            theta = np.arctan2(p2[0] - cx, p2[1] - cy)
            s = np.linspace(theta, theta - np.sign(self.dor_vec[1])*np.pi, 400)

        elif (self.animal_x_center - self.anterior[0] > 0):
            theta = np.arctan2(p1[0] - cx, p1[1] - cy)
            s = np.linspace(theta, theta - np.sign(self.dor_vec[1])*np.pi, 400)

        x = cy + int(diameter/2)*np.cos(s)
        y = cx + int(diameter/2)*np.sin(s)

        self.pedestal_snake =  np.array([x, y]).T
    
    def fit_pedestal(self, im, hc=True, npoints=200, bound=0.2, ma=4, prune_ma=4, prune_threshold=3):

        if self.pedestal_snake is None: self.initialize_snake()
        ps = self.pedestal_snake

        head = self.head
        tail = self.tail
        dp = self.dorsal_point
        
        w, h = im.shape

        if hc:
            # we use Contrast Limited Adaptive Histogram Equalization to 
            # increase contrast around pedestal for better edge detection
            
            bb_x = ( int(np.max([np.min(ps[:,1]), 0])), int(np.min([np.max(ps[:,1]), w]) ))
            bb_y = ( int(np.max([np.min(ps[:,0]), 0])), int(np.min([np.max(ps[:,0]) ,h]) ))

            cropped = im[bb_x[0]:bb_x[1], bb_y[0]:bb_y[1]]
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
            result = clahe.apply(cropped)
            im[bb_x[0]:bb_x[1], bb_y[0]:bb_y[1]] = result
        
        dot = 4*self.gradient(im, "dorsal", sigma=1) + self.gradient(im, "anterior", sigma=1)
        
        n = ps.shape[0]

        if self.dist(head, (ps[0,1], ps[0,0]) ) > 2:
            ps = np.flip(ps, 0)
        
        snakex = ps[:,0]
        snakey = ps[:,1]

        x1 = snakex[0]
        y1 = snakey[0]
        x2 = snakex[-1]
        y2 = snakey[-1]
        
        m = (y2 - y1)/(x2 - x1)
        b = y1 - m*x1

        m2 = -1/m

        ped_x = []
        ped_y = []

        edge = []
        pruned_edge = []
        init = []
        for i in xrange(n-1):
            
            p2 = snakex[i], snakey[i]
            b2 = p2[1] - m2*p2[0]

            x = (b2 - b)/(m - m2)
            y = (m2*x + b2)
            p1 = x, y
            
            e = self.find_edge(dot, p2, p1, npoints, ma, w_threshold=20)
            if e is not None:
                init.append((x,y))
                edge.append(e)
        
        window = int(prune_ma/2)
        edge = np.array(edge)
        for i in xrange(window, edge.shape[0] - window):
            avg = np.mean(edge[i-window:i+window, :], axis=0)
            if self.dist(edge[i, :], avg) < prune_threshold:
            #    print edge[i,:], init[i]
                pruned_edge.append((i, self.dist(edge[i, :], init[i])))
        
        #pruned_edge_normalized = [pruned_edge[:,0], pruned_edge[:,1]/(self.pixel_to_mm/self.animal_length)]
        return pruned_edge
    
    def get_pedestal_max_height(self, data):
        
        self.pedestal_max_height = np.max(data[:,1])

    def get_pedestal_area(self, data):
        print data.shape 
        self.pedestal_area = np.sum(0.5*(self.dist(self.head, self.dorsal_point)/400)*(data[1:][:,0] - data[0:-1][:,0])*(data[1:][:,1] + data[0:-1][:,1]))
        
    def get_pedestal_theta(self, data, n=200):
        
        x = (n - data[np.argmax(data[:,1]), 0]) * self.dist(self.head, self.dorsal_point)/400
        hyp = self.dist((n,0), (x, np.max(data[:,1])))
        self.pedestal_theta = np.arcsin((n - x)/hyp)*(180/np.pi)

    def find_edge(self, im, p1, p2, npoints=400, ma=4, bound=0.2, w_threshold=50):

        # x and y in p1 and p2 are ordered in image convention, but map_coordinates is in ordinal
        xx, yy = np.linspace(p1[1], p2[1], npoints), np.linspace(p1[0], p2[0], npoints)

        zi = scipy.ndimage.map_coordinates(im, np.vstack((xx, yy)), mode='nearest')
        zi -= np.mean(zi)
        
        zi = pd.rolling_mean(zi, ma)

        lb = np.nanmean(zi) - bound*(np.nanmean(zi) - np.nanmin(zi))
        ub = np.nanmean(zi) + bound*(np.nanmax(zi) - np.nanmean(zi))
        
        mins = []
        maxes = []

        for i in xrange(ma, len(zi)-1):
            if (zi[i] - zi[i-1] < 0) and (zi[i+1] - zi[i] > 0) and (zi[i] < lb):
                mins.append(i)
            elif (zi[i] - zi[i-1] > 0) and (zi[i+1] - zi[i] < 0) and (zi[i] > ub):
                maxes.append(i)
	
        for j in mins:
            for k in maxes:
                if np.abs(j - k) < w_threshold:
                    return (yy[j], xx[j])          # intentionally return first trough
        return

    def find_edge2(self, im, p1, p2, t=0.1, npoints=400):

        xx, yy = np.linspace(p1[1], p2[1], npoints), np.linspace(p1[0], p2[0], npoints)
        zi = scipy.ndimage.map_coordinates(im, np.vstack((yy, xx)), mode="nearest")
        zi = pd.rolling_mean(zi, 4)

        for i in xrange(len(zi)):
            if zi[i] > t:
                return (yy[i], xx[i])

    def gradient(self, im, direction, sigma=0.5):

        dx, dy = np.gradient(gaussian(im, sigma))

        dir_x, dir_y = getattr(self, direction[0:3] + "_vec")

        return dir_x*dx + dir_y*dy

    def high_contrast(self, im):
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(im)
    
    def area(self, x, y):

        x = np.asanyarray(x)
        y = np.asanyarray(y)
        n = len(x)

        up = np.arange(-n+1, 1)
        down = np.arange(-1, n-1)

        return (x * (y.take(up) - y.take(down))).sum() / 2
    def intersect(self, s1, s2):

        x1, y1, x2, y2 = s1
        x3, y3, x4, y4 = s2

        if (max([x1, x2]) < min([x3, x4])): return False

        m1 = (y1 - y2)/(x1 - x2)
        m2 = (y3 - y4)/(x3 - x4)

        if (m1 == m2): return False
        
        b1 = y1 - m1*x1
        b2 = y3 - m2*x3

        xa = (b2 - b1) / (m1 - m2)

        if ( (xa < max( [min([x1, x2]), min([x3, x4])] )) or (xa > min( [max([x1, x2]), max([x3, x4])] )) ):
            return False

        return True

    def intersection(self, s1, s2):

        # returns the point of intersection for two line segments

        if not self.intersect(s1, s2): return None

        x1, y1, x2, y2 = s1
        x3, y3, x4, y4 = s2

        return (((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/
                ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)),
                ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/
                ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))

    def norm( self, x ):

        return (x - np.min(x)) / (np.max(x) - np.min(x))
