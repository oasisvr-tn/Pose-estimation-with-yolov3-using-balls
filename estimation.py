#class id 32 for balls
from ctypes import sizeof
import cv2
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import plotly.graph_objects as go
import socket
import time
import warnings
import threading
import os
prev=0
new=0
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 25003))
R1 = np.array([[-0.1434, 0.9896, 0.0092],
              [0.1672, 0.0334, -0.9854],
              [-0.9754, -0.1397, -0.1702]])
R1_inv = np.linalg.inv(R1)
T1 = np.array([[-88.1, 804.8, 4315.3]]).T
R2 = np.array([[0.9654, 0.2600, -0.0214],
              [0.0151, -0.1372, -0.9904],
              [-0.2605, 0.9558, -0.1364]])
R2_inv = np.linalg.inv(R2)
T2 = np.array([[206.1, 955.7, 4062.3]]).T
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
pTime = 0
camera1 = cv2.VideoCapture(r"C:\detectiongpu\1.avi")
camera2 = cv2.VideoCapture(r"C:\detectiongpu\2.avi")
kpt_3D=[[0]]
while True:
    _,img1=camera1.read()
    _,img2=camera2.read()
    height1,width1,_=img1.shape
    height2,width2,_=img2.shape
    blob1=cv2.dnn.blobFromImage(img1,1.0/255,(320,320),(0, 0,0),True,crop=False)
    blob2=cv2.dnn.blobFromImage(img2,1.0/255,(320,320),(0, 0,0),True,crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)
    class_ids1 = []
    confidences1 = []
    boxes1 = []
    centers1=[]
    for out1 in outs1:
        for detection1 in out1:
            scores1 = detection1[5:]
            class_id1 = np.argmax(scores1)
            if class_id1 == 0:
                confidence1 = scores1[class_id1]
                if confidence1 > 0.5:
                    center_x1 = int(detection1[0] * width1)
                    center_y1 = int(detection1[1] * height1)
                    w1 = int(detection1[2] * width1)
                    h1 = int(detection1[3] * height1)
                    x1 = int(center_x1 - w1 / 2)
                    y1 = int(center_y1 - h1 / 2)
                    boxes1.append([x1, y1, w1, h1])
                    confidences1.append(float(confidence1))
                    class_ids1.append(class_id1)      
    indexes1 = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i1 in range(len(boxes1)):
        if i1 in indexes1:
            x1, y1, w1, h1 = boxes1[i1]
            label1 = str(classes[class_ids1[i1]])
            color1 = colors[i1]
            centers1.append([int(x1+w1/2),int(y1+h1/2)])
            cv2.circle(img1,(int(x1+w1/2),int(y1+h1/2)), 1, (0,0,255), -1)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)
    class_ids2 = []
    confidences2 = []
    boxes2 = []
    centers2=[]
    for out2 in outs2:
        for detection2 in out2:
            scores2 = detection2[5:]
            class_id2 = np.argmax(scores2)
            if class_id2 == 0:
                confidence2 = scores2[class_id2]
                if confidence2 > 0.5:
                    center_x2 = int(detection2[0] * width2)
                    center_y2 = int(detection2[1] * height2)
                    w2 = int(detection2[2] * width2)
                    h2 = int(detection2[3] * height2)
                    x2 = int(center_x2-w2/2)
                    y2 = int(center_y2-h2/2)
                    boxes2.append([x2, y2, w2, h2])
                    confidences2.append(float(confidence2))
                    class_ids2.append(class_id2)      
    indexes2 = cv2.dnn.NMSBoxes(boxes2, confidences2, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i2 in range(len(boxes2)):
        if i2 in indexes2:
            x2, y2, w2, h2 = boxes2[i2]
            label2 = str(classes[class_ids2[i2]])
            color2 = colors[i2]
            centers2.append([int(x2+w2/2),int(y2+h2/2)])
            cv2.circle(img2,(int(x2+w2/2),int(y2+h2/2)), 1, (0,0,255), -1)
    y1,x1=centers1[0][0],centers1[0][1]
    y2,x2=centers2[0][0],centers2[0][1]
    y1_d,x1_d=y1-T1[1],x1-T1[0]
    y2_d,x2_d=y2-T2[1],x2-T2[0]
    R_d=np.array([R1_inv[:,0],R1_inv[:,1],R2_inv[:,0],R2_inv[:,1]]).T
    Y=np.dot(R_d,np.array([[-1*x1_d[0],-1*y1_d[0],x2_d[0],y2_d[0]]]).T)
    X=np.array([R1_inv[:,2],R2_inv[:,2]]).T
    W=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
    z1_d=W[0,0]
    kpt_W=np.dot(R1_inv,np.array([[x1_d,y1_d,z1_d]]).T)
    kpt_3D[0]=[kpt_W[0,0][0],kpt_W[1,0][0],kpt_W[2,0][0]]
    posString =str(kpt_3D[0])
    sock.sendall(posString.encode("UTF-8"))
    new= time.time()
    fps = 1/(new-prev)
    print(fps)
    prev=new
    if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()