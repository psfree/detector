#!/usr/bin/python
from imutils import paths
import numpy as np
import cv2
import struct
import serial
from PID import PID


def variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def intersection(rect1, rect2):
    try:
        p1 = Polygon([(rect1[0], rect1[1]), (rect1[1], rect1[1]),
                     (rect1[2], rect1[3]), (rect1[2], rect1[1])])
        p2 = Polygon([(rect2[0], rect2[1]), (rect2[1], rect2[1]),
                     (rect2[2], rect2[3]), (rect2[2], rect2[1])])
        return p1.intersects(p2)
    except:
        return True


# ser1 = serial.Serial('/dev/cu.usbmodem143301', 9600) #set the correct port
##ser1.write('f'.encode())

faceCascade = cv2.CascadeClassifier('frontal.xml')
sideCascade = cv2.CascadeClassifier('profile.xml')
upperbodyCascade = cv2.CascadeClassifier('upperbody.xml')

# cap = cv2.VideoCapture('bus.mp4')

cap = cv2.VideoCapture(0)

pid = PID(1, 0.1, 0.05, setpoint=1000) #tweak me. hopefully setpoint being high will not cause problems
pid.sample_time = 0.05 #20fps or about double the expected sampling time. may need tweaking

while cap.isOpened():
    (ret, frame) = cap.read()
    if ret == False:
        continue

    # frame =  cv2.blur(frame, (100,100))
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
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(100, 100),
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
    
    for a in faces:
        for b in profiles:
            if intersection(a, b):
                sections.append(b)
        for c in fprofiles:
            if intersection(a, c):
                sections.append(c)
        for d in bodies:
            if intersection(a, d):
                sections.append(d)
        if len(sections) > 0:
        	sections.append(a)
        	break
    cx = 0
    cy = 0
    for (x, y, w, h) in sections:
        cx = (cx + x + w / 2) / 2
        cy = (cy + y + h / 2) / 2
	
    #print(cx)
    (height, width, channels) = frame.shape
    if len(faces) == 0:
        zzzz=1
        # ser1.write('cc\n'.encode())

        #print ('cc\n')
    fudge = 50
    
    
	
    for (x, y, w, h) in faces:  #does not currently use intersection data
        pos1 = x + w / 2
        pos2 = y + h / 2
        s1 = b'c'
        s2 = b'c'
        if pos2 < height / 2 - fudge:
            s1 = b'u'
        elif pos2 > height / 2 + fudge:
            s1 = b'd'

        if pos1 < width / 2 - fudge:
            s2 = b'r'
        elif pos1 > width / 2 + fudge:
            s2 = b'l'

        s = struct.pack("<ccfc", s1,s2,control, b'\n')
        # ser1.write(s.encode())

        print(s)
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

			