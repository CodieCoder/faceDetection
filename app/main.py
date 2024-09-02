import cv2
import cv2.data
import matplotlib.pyplot as plt

imagePath = './mock/file.png'
img = cv2.imread(imagePath)#Read the image with openCV
img_in_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Load the built-in pre-trained Haar Cascade classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#now for face detection using our above classifier
face = face_classifier.detectMultiScale(img_in_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

#drawing a bounding box around detected faces
for(x, y, w, h) in face:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0),thickness=4)      
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#convert from BGR to RGB

#Use Matplotlib to display the image and the bounding boxes
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

# print(face)