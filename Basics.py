import numpy
import cv2
import face_recognition
imgElon = face_recognition.load_image_file('Elon Musk main.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
cv2.startWindowThread()
imgElontest = face_recognition.load_image_file('Elon Musk test.jpg')
#imgElontest = face_recognition.load_image_file('Bill_Gates test.jpg')
imgElontest = cv2.cvtColor(imgElontest,cv2.COLOR_BGR2RGB)
# cv2.imshow('Elon Musk main',imgElon)
# cv2.imshow('Elon Musk test',imgElontest)
# cv2.waitKey(0)
faceLoc = face_recognition.face_locations(imgElon)[0]
#print(faceLoc)
encodeElon = face_recognition.face_encodings(imgElon)[0] 
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# cv2.imshow('Elon Musk main',imgElon)
# cv2.waitKey(0)
faceLoctest = face_recognition.face_locations(imgElontest)[0]
encodeElontest = face_recognition.face_encodings(imgElontest)[0] 
cv2.rectangle(imgElontest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)
# cv2.imshow('Elon Musk main',imgElon)
# cv2.imshow('Elon Musk test',imgElontest)
# cv2.waitKey(0)
results = face_recognition.compare_faces([encodeElon],encodeElontest)
print(results)
faceDis = face_recognition.face_distance([encodeElon],encodeElontest)
print(faceDis)
cv2.putText (imgElontest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1 ,(0,0,255),2)
cv2.imshow('Elon Musk test',imgElontest)
cv2.waitKey(0)