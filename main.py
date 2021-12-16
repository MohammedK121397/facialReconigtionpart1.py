import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to scan --> this is how you important an image using cv
#mg = cv2.imread('venv/ionman.jpg')
img = cv2.imread('venv/example2detector.png') # go to video time 55min for this 3 face detector| known as sliding window

#colour image now will be turned into grey , black and white to allow the scan (Grey sScale)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces                   this is looking for the face composition. whether it gets small or big. it will detect it
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#print(face_coordinates)

#draw rectangles around faces + do a for loop. saves having to repeat code.
for (x, y, w, h) in face_coordinates:
#(x, y, w, h) = face_coordinates[0]
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)

#(x,y), (x+w), (y+h),35  78 258 258
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print("Code compelted.")