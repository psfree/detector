#!/usr/bin/python
# -*- coding: utf-8 -*-
import pdb
from imutils import paths
import numpy as np
import cv2
import struct
import serial
import serial.tools.list_ports
import sys
import datetime
from time import sleep
from PID import PID

GREEN = (0, 0xFF, 0)
BLUE = (0xFF, 0, 0)
RED = (0, 0, 0xFF)


##draw_boxes: used to draw a rectangle around tracked features for debugging purposes

def draw_boxes(frame, boxes, color=(0, 0xFF, 0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y
                      + h)), color, 2)
    return frame


class BodyDetector:

    def __init__(self, cascPath='upperbody.xml'):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.faceCascade.detectMultiScale(gray,
                scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        return bodies


class ProfileDetector:

    def __init__(self, flip=False, cascPath='profile.xml'):
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.flip = flip

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width, channels) = frame.shape
        if self.flip:
            grayflip = gray.copy()
            grayflip = cv2.flip(gray, 1)
            gray = grayflip
        profile = self.faceCascade.detectMultiScale(gray,
                scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))
        if self.flip:
            for i in profile:
                i[0] = width - i[0] - i[2]
        return profile


class FaceDetector:

    def __init__(self, cascPath='frontal.xml'):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,
                scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))
        return faces


class AnyTracker:

    def __init__(self, frame, rect):
        (x, y, w, h) = rect
        self.rect = (x, y, w, h)

        # can select alternative tracking algorithm here

        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.rect)

    def update(self, frame):
        (_, self.rect) = self.tracker.update(frame)
        return self.rect


class Monitor:

    def __init__(self, interval_time=5):
        self.interval = interval_time
        self.previous = datetime.datetime.now()

    def trigger(self):
        return self.getTimeDiff() > self.interval

    def getTimeDiff(self):
        return (datetime.datetime.now() - self.previous).seconds

    def restart(self):
        self.previous = datetime.datetime.now()


class Tracking:

    def __init__(self, interval_time=5):
        self.monitor = Monitor(interval_time)
        self.face_detect = FaceDetector()
        self.prof_detect = ProfileDetector()
        self.fprof_detect = ProfileDetector(flip=True)
        self.body_detect = BodyDetector()
        self.tracklist = []
        self.cx = 0
        self.cy = 0

    def detect_track(self, frame):
        faces = self.face_detect.detect(frame)
        profiles = self.prof_detect.detect(frame)
        fprofiles = self.fprof_detect.detect(frame)
        bodies = self.body_detect.detect(frame)
        sections = []
        self.tracklist=[]
        if len(faces)==0:
            faces = profiles
            if len(profiles)==0:
                faces = fprofiles
                if len(fprofiles)==0:
                    faces=bodies
                    bodies=[]
                fprofiles=[]
            profiles = []
        facelist = []
        casclists = []
        for a in faces:
            sections = []
            for b in profiles:
                if intersection(a, b):
                    sections.append(b)
            for c in fprofiles:
                if intersection(a, c):
                    sections.append(c)
            for d in bodies:
                if intersection(a, d):
                    sections.append(d)
            sections.append(a)
            facelist.append(a)
            casclists.append(sections)
        if len(casclists) > 0:
            ind=0
            max=0
            for each in enumerate(casclists):
                sz = len(each[1])
                if sz>max:
                    max = sz
                    ind = each[0]
            casc_sel = casclists[ind]
            self.cx = 0
            self.cy = 0
            for (x, y, w, h) in casc_sel:
                self.cx = (self.cx + x + w / 2) / 2
                self.cy = (self.cy + y + h / 2) / 2
            for f in casc_sel:
                self.tracklist.append(AnyTracker(frame, f))
            faces = [facelist[ind]]
        else:
            self.tracklist = []
        self.monitor.restart()
        new = type(faces) is not tuple
        return (faces, new)

    def track(self, frame):
        boxez = []
        for each in self.tracklist:
            boxez.append(each.update(frame))
        self.cx = 0
        self.cy = 0
        for (x,y,w,h) in boxez:
        	self.cx = (self.cx + x + w / 2) / 2
        	self.cy = (self.cy + y + h / 2) / 2
        if len(boxez) > 0:
            boxes = [boxez[-1]]
        else:
        	boxes = boxez
        return (boxes, False)

    def boxframes(self, frame):
        if self.monitor.trigger():
            return self.detect_track(frame)
        else:
            return self.track(frame)


# variance: this function calculates the variance of the laplacian of an image frame
# which is useful for detecting the amount of blur in the image

def variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# intersection: this function is used to calculate whether 2 rects intersect each other
# which is useful for checking if different tracking features are detecting the same person

def intersection(rect1, rect2):
    try:
        p1 = Polygon([(rect1[0], rect1[1]), (rect1[1], rect1[1]),
                     (rect1[2], rect1[3]), (rect1[2], rect1[1])])
        p2 = Polygon([(rect2[0], rect2[1]), (rect2[1], rect2[1]),
                     (rect2[2], rect2[3]), (rect2[2], rect2[1])])
        return p1.intersects(p2)
    except:
        return True


# this is the main body of the code
# to start we begin by enumerating the serial ports that are available and allowing the
# user to select which one to use

def main():
    ports = list(serial.tools.list_ports.comports())
    
    i = 0
    for p in ports:
        print(str(i) + ': ' + str(p))
        i = i + 1
    while True:
        sel = input('Select a number from the list to choose the Arduino port: ')
        try:
            j = int(sel)
            if j < 0 or j > len(ports)-1:
                print('Invalid number selected, try again')
            else:
                break
        except ValueError:
            print('Please choose a number from the list')
    portchoice = str(ports[j].device)
    
    # ser1 = serial.Serial(portchoice, 9600) #set the correct port
    
    # open the image capture device
    
    cap = cv2.VideoCapture(1)
    (ret, frame) = cap.read()

    event_interval = 1.5
    
    tracking = Tracking(event_interval)
    
    # create  a PID controller for use with zoom and focus
    
    pid = PID(1, 0.1, 0.05, setpoint=1000)  # tweak me. hopefully setpoint being high will not cause problems
    pid.sample_time = 0.05  # 20fps or about double the expected sampling time. may need tweaking
    
    # main loop of the code. this reads a frame in, processes it and performs the 4 tracking operations
    # it ends by sending the relevant data to the arduino over the serial port
    
    storetime = datetime.datetime.now()
    framecount=0
    fps = 20
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == False:
            continue
        framecount+=1
        if(datetime.datetime.now()- storetime).seconds > 1:
            storetime = datetime.datetime.now()
            fps = framecount
            framecount =0
            
            
    
        (boxes, detect_new) = tracking.boxframes(frame)
        color = (GREEN if detect_new else BLUE)
        draw_boxes(frame, boxes, color)
    
        # frame =  cv2.blur(frame, (100,100))
    
        (height, width, channels) = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = variance(gray)
    
        control = pid(fm)
        
        # control the motors for autofocus here
        # send control output to the arduino
    
        text = 'Blur'
        cv2.putText(
            frame,
            '{}: {:.2f}'.format(text, fm),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0xFF),
            3,
            )
            
        text = 'FPS'
        cv2.putText(
            frame,
            '{}: {:.2f}'.format(text, fps),
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0xFF),
            3,
            )
        fudge=30
        cx = tracking.cx
        cy = tracking.cy
        s1 = b'c'
        s2 = b'c'
        if cy < (height / 2 - fudge):
            s1 = b'u'
        elif cy > (height / 2 + fudge):
            s1 = b'd'
    
        if cx < width / 2 - fudge:
            s2 = b'r'
        elif cx > width / 2 + fudge:
            s2 = b'l'
    	
        s = struct.pack("<ccfc", s1,s2,control, b'\n')
        #ser1.write(s)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # gracefully close the camera capture when exiting
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
