import cv2
import numpy as np 
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

'''
3 Haar Cascades




'''


'''
2 Creating video capture
cap = cv2.VideoCapture(0)
fourcc = cvw.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	out.write(frame)
	cv2.imshow('frame', frame)
	cv2.imshow('gray', gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()'''

#img = cv2.imread('./assets/cat.jpeg', cv2.IMREAD_GRAYSCALE)

'''
1 Creating image capture
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50,100],[80,100], 'c', linewidth=5)
plt.show()'''

#cv2.imwrite('greycat.png', img)