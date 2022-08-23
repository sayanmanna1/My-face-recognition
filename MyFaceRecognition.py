import cv2
import numpy as np
import face_recognition

def conclude(value, dist):
    if value[0] == True and dist[0] < 0.5:
        return 1
    else:
        return 0
cap = cv2.VideoCapture(0) 
success, img = cap.read()
cap.release()
cv2.destroyAllWindows() 
imgMy = face_recognition.load_image_file('your_image.jpeg')
imgMy = cv2.cvtColor(imgMy,cv2.COLOR_BGR2RGB)
imgTest = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(imgMy)[0]
encodeMy = face_recognition.face_encodings(imgMy)[0]
cv2.rectangle(imgMy,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
results = face_recognition.compare_faces([encodeMy],encodeTest)
faceDis = face_recognition.face_distance([encodeMy],encodeTest)#less the distance more the match
print(results)
finalValue = conclude(results,faceDis) 
if finalValue==1:
    cv2.putText(imgTest,'True',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
else:
    cv2.putText(imgTest,'False',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Face",imgTest)
# cv2.imshow("My",imgMy)
cv2.waitKey(0)


