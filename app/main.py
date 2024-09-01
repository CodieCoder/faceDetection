import cv2
import matplotlib.pyplot as plt

try:
  imagePath = 'file.png'
  img = cv2.imread(imagePath)  # Read the image with openCV

  if img is None:
    raise Exception("Error reading image file")

  img_in_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Load the built-in pre-trained Haar Cascade classifier
  face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

  # Face detection using the classifier
  face = face_classifier.detectMultiScale(img_in_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

  # Drawing bounding boxes around detected faces
  for (x, y, w, h) in face:
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

  # Display the image with bounding boxes using OpenCV (optional)
  # cv2.imshow('Image with Detected Faces', img)
  # cv2.waitKey(0)  # Wait for a key press to close the window

  # Convert to RGB for Matplotlib display (if needed)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Use Matplotlib to display the image and the bounding boxes (optional)
  plt.figure(figsize=(20, 10))
  plt.imshow(img_rgb)
  plt.axis('off')

  # Print the detected face coordinates (optional)
#   print(face)

  plt.show()  # Display the Matplotlib plot

except Exception as e:
  print(f"Error: {e}")