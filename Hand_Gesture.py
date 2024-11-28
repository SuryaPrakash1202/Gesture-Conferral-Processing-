import os

import cv2

from cvzone.HandTrackingModule import HandDetector

import numpy as np

width,height = 1200, 720

folderPath = "presentation"

#camera setup

cap = cv2.VideoCapture(0)

cap.set(3,width)

cap.set(4,height)

#get the list of presentation images

pathImages = sorted(os.listdir(folderPath),key=len)

#print(pathImages)

#variables

imgNumber = 0

hs, ws = int(120*1.2), int(213*1)

gestureThreshold = 300

buttonPressed = False

buttonCounter = 0

buttonDelay = 30
annotations = [[]]

annotationNumber = 0

annotationStart = False

#hand detector

detector = HandDetector(detectionCon=0.8,maxHands=1)

while True:

#import images

success,img = cap.read()

pathFullImage = os.path.join(folderPath,pathImages[imgNumber])

imgCurrent = cv2.imread(pathFullImage)

hands, img = detector.findHands(img,flipType=False)

cv2.line(img,(0,gestureThreshold),(width,gestureThreshold),(0,255,0),10)

if hands and buttonPressed is False:

hand = hands[0]

fingers = detector.fingersUp(hand)

cx,cy = hand['center']

print(fingers)

lmList = hand['lmList']

#constrain values for eaiser drawing

xVal = int(np.interp(lmList[0][0],[width//2,ws],[0,width]))

yVal = int(np.interp(lmList[0][1],[150,height-150], [0, height]))

indexFinger = xVal,yVal
if cy <=gestureThreshold: #if hand is at the height of the face

annotationStart = False

#gesture 1 =left

if fingers == [0,0,0,0,0]:

annotationStart = False

print("Left")

if imgNumber>0:

buttonPressed = True

annotations = [[]]

annotationNumber = 0

imgNumber -= 1

if fingers == [1, 0, 0, 0, 1]:

annotationStart = False

print("Right")

buttonPressed = True

if imgNumber < len(pathImages)-1:

buttonPressed = True

annotations = [[]]

annotationNumber = 0

imgNumber += 1

#Gesture 3 = show pointer

if fingers == [1,1,1,0,0]:
  cv2.circle(imgCurrent, indexFinger,12,(0,0,255),cv2.FILLED)

annotationStart = False

#draw pointer

if fingers == [1,1,0,0,0]:

if annotationStart is False:

annotationStart = True

annotationNumber +=1

annotations.append([])

cv2.circle(imgCurrent, indexFinger,12,(0,0,255),cv2.FILLED)

annotations[annotationNumber].append(indexFinger)

else:

annotationStart = False

#gesture 5 - erase

if fingers == [1,1,1,1,0]:

if annotations:

if annotationNumber>= 0:

annotations.pop(-1)

annotationNumber -=1

buttonPressed = True

else:

annotationStart = False

#button pressed iterations
if buttonPressed:

buttonCounter += 1

if buttonCounter>buttonDelay:

buttonCounter = 0

buttonPressed = False

for i in range (len(annotations)):

for j in range(len(annotations[i])):

if j!=0:

cv2.line(imgCurrent,annotations[i][j-1],annotations[i][j],(0,0,200),12)

#adding webcam images on the slides

imgsmall = cv2.resize(img,(ws,hs))

h,w,_ = imgCurrent.shape

imgCurrent[0:hs,w-ws:w] = imgsmall

cv2.imshow("Image", img)

cv2.imshow("slides",imgCurrent)

key = cv2.waitKey(1)

if key == ord('q'):

break

import cv2

import numpy as np

import face_recognition

import os
from datetime import datetime

path ='images'

images =[]

classnames =[]

mylist =os.listdir(path)

print(mylist)

for cls in mylist:

curimg =cv2.imread(f'{path}/{cls}')

images.append(curimg)

classnames.append(os.path.splitext(cls)[0])

print(classnames)

def findencodings(images):

encodelist =[]

for img in images:

img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

encode =face_recognition.face_encodings(img)[0]

encodelist.append(encode)

return encodelist

def markattendence(name):

with open('Attendance.csv','r+')as f:

mydatalist =f.readlines()

namelist =[]
for line in mydatalist:

entry =os.path.split(',')

namelist.append(entry[0])

if name not in mydatalist:

now =datetime.now()

dtstring =now.strftime('%H:%M:%S')

f.writelines(f'\n{name},{dtstring}')

newface = findencodings(images)

print(len(newface))

cap =cv2.VideoCapture(0)

while True:

sucess,img =cap.read()

imgs =cv2.resize(img,(0,0),None,0.25,0.25)

imgs =cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

faceencode =face_recognition.face_encodings(imgs)

faceloc =face_recognition.face_locations(imgs)

for encodeface,locface in zip(faceencode,faceloc):

matches=face_recognition.compare_faces(newface,encodeface)

facedis =face_recognition.face_distance(newface,encodeface)

print(facedis)

matchindex =np.argmin(facedis)

if matches[matchindex]:
  name =classnames[matchindex].upper()

print(name)

y1,x2,y2,x1 =locface

y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4

cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

cv2.putText(img,name,(x1+6,y2-

6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

cv2.imshow('webcam',img)

cv2.waitKey(1)
