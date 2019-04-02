#!/usr/bin/python
import imutils
from imutils import paths
import numpy as np
import cv2
import struct
import serial
import serial.tools.list_ports
import datetime
from PID import PID

#variance: this function calculates the variance of the laplacian of an image frame
# which is useful for detecting the amount of blur in the image
def variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

#intersection: this function is used to calculate whether 2 rects intersect each other
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

#this is the main body of the code
#to start we begin by enumerating the serial ports that are available and allowing the
#user to select which one to use
ports = list(serial.tools.list_ports.comports())

i=0
for p in ports:
    print(str(i)+": "+str(p))
    i=i+1
while True:
    sel = input("Select a number from the list to choose the Arduino port: ")
    try:
        j = int(sel)
        if j<0 or j>len(ports)-1:
            print("Invalid number selected, try again")
        else:
            break
    except ValueError:
        print("Please choose a number from the list")
portchoice = str(ports[j].device)
ser1 = serial.Serial(portchoice, 9600) #set the correct port

#load the tracking data from the xml files
faceCascade = cv2.CascadeClassifier('frontal.xml')
sideCascade = cv2.CascadeClassifier('profile.xml')
upperbodyCascade = cv2.CascadeClassifier('upperbody.xml')


#open the image capture device
cap = cv2.VideoCapture(0)

#create  a PID controller for use with zoom and focus
pid = PID(1, 0.1, 0.05, setpoint=1000) #tweak me. hopefully setpoint being high will not cause problems
pid.sample_time = 0.05 #20fps or about double the expected sampling time. may need tweaking

storetime = datetime.datetime.now()
framecount=0
fps = 20
#main loop of the code. this reads a frame in, processes it and performs the 4 tracking operations
#it ends by sending the relevant data to the arduino over the serial port
while cap.isOpened():
    (ret, frame) = cap.read()
    if ret == False:
        continue
    framecount+=1
    if(datetime.datetime.now()- storetime).seconds > 1:
        storetime = datetime.datetime.now()
        fps = framecount
        framecount =0
    # frame =  cv2.blur(frame, (100,100))
    frame = imutils.resize(frame, width=720)
    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance(gray)
    
    control = pid(fm)
    #print(control)
    
    #control the motors for autofocus here
    #send control output to the arduino
    
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
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=10, minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE)
    profiles = sideCascade.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)

    bodies = upperbodyCascade.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)
    grayflip = gray.copy()
    grayflip = cv2.flip(gray, 1)
    fprofiles = sideCascade.detectMultiScale(grayflip, scaleFactor=1.1,
            minNeighbors=5, minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xFF, 0), 2)
    for (x, y, w, h) in profiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0xFF, 0xFF, 0), 2)
    for (x, y, w, h) in fprofiles:
        cv2.rectangle(frame, (width-x, y), (width-(x+w), y + h), (0, 0, 0), 2)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0xFF, 0, 0), 2)

    sections = []
    if len(faces)==0:
    	faces = profiles
    	if len(profiles)==0:
    		faces = fprofiles
    		if len(fprofiles)==0:
    			faces=bodies
    			bodies=[]
    		fprofiles=[]
    	profiles = []
    cx = width/2
    cy = height/2
    for (x,y,w,h) in faces:
    	cx = x + w/2
    	cy= y+h/2
    	break
    fudge=30
    s1 = b'c'
    s2 = b'c'
    if cy < (height / 2 - fudge):
        s1 = b'd'
    elif cy > (height / 2 + fudge):
        s1 = b'u'

    if cx > (width / 2 + fudge):
        s2 = b'l'
    elif cx < (width / 2 - fudge):
        s2 = b'r'
	
    print(str(s1))
    s = struct.pack("<ccfc", s1,s2,control, b'\n')
    ser1.write(s)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#gracefully close the camera capture when exiting
cap.release()
cv2.destroyAllWindows()

			