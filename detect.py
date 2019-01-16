from imutils import paths
import numpy as np
import cv2
import serial

def variance(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

ser1 = serial.Serial('/dev/cu.usbmodem143301', 9600) #set the correct port
#ser1.write('f'.encode())

faceCascade = cv2.CascadeClassifier('frontal.xml')
sideCascade = cv2.CascadeClassifier('profile.xml')

#cap = cv2.VideoCapture('bus.mp4')
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame =  cv2.blur(frame, (100,100))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance(gray)
    text = "Blur"
    cv2.putText(frame, "{}: {:.2f}".format(text, fm), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    profiles = sideCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in profiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    height, width, channels = frame.shape
    if(len(faces)==0):
    	ser1.write('c'.encode())
    	print("c")
    fudge = 50
    for (x,y,w,h) in faces:
        pos1  = x+w/2
        pos2  = y+h/2
        s1 = 'c'
        s2 = 'c'
        if(pos2 < height/2 -fudge):
             s1 = 'u'
        elif(pos2 > height/2 + fudge):
             s1 = 'd'
             
        if(pos1 < width/2 -fudge):
             s2 = 'r'
        elif(pos1 > width/2 + fudge):
             s2= 'l'
        s= s1+s2 +'\n';
        ser1.write(s.encode())
        print(s.encode())
        break
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
