import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(videoFeed):
    grey_image = cv2.cvtColor(videoFeed, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_image, 1.1, 5, minSize=(40, 40))
    for(x, y, w, h) in faces:
        cv2.rectangle(videoFeed, (x, y), (x +w, y + h), (0, 255, 0), 4)
    
    return faces


while True:
    #read from the video
    result, videoFeed = video_capture.read()
    
    #terminate the loop if the frame is not read successfully
    if result is False:
        print("An error occurred. Video stream could not be read.")
        break
    
    #apply the function to display the bounding box around detected images
    faces = detect_bounding_box(videoFeed=videoFeed)

    #Display the processed frame in a window named "My face detection Project"
    cv2.imshow("My face Detection Project ", videoFeed)\
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.destroyAllWindows()
        break
 

video_capture.release()
cv2.destroyAllWindows()
