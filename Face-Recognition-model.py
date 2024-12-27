import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "D:\\village\\Face-Recognition-model\\Training_images"
images = []
classNames = []

attendance_directory = "D:\\village\\Face-Recognition-model\\Attendance_Records\\"  # Modify this to your preferred directory

if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

attendance_file_path = os.path.join(attendance_directory, "Attendance.csv")

attendance_dict = {}

myList = os.listdir(path)
print("Image list:", myList)

for cl in myList:
    curImg = cv2.imread(f'{path}\\{cl}')
    if curImg is None:
        print(f"Failed to load image {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

attendance_dict = {name.upper(): 'Absent' for name in classNames}
print("Class names:", classNames)

def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        if img is None:
            print(f"Image at index {i} is None")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        
        if len(encodes) == 0:
            print(f"No faces found in image at index {i}")
            continue
        
        encodeList.append(encodes[0])
    
    return encodeList

def createCSV():
    if not os.path.isfile(attendance_file_path):
        with open(attendance_file_path, 'w') as f:
            f.write('Name,Time,Status\n')

def updateAttendanceCSV(attendance_dict):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    with open(attendance_file_path, 'a') as f:
        for name, status in attendance_dict.items():
            f.write(f'{name},{dtString},{status}\n')

encodeListKnown = findEncodings(images)
print('Encodings Complete')

createCSV()

cap = cv2.VideoCapture(0)

present_set = set()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if name not in present_set:  
                print(f"Recognized {name}")  
                attendance_dict[name] = 'Present'  
                present_set.add(name)  

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for name in classNames:
    if name.upper() not in present_set:
        attendance_dict[name.upper()] = 'Absent'
updateAttendanceCSV(attendance_dict)

cap.release()
cv2.destroyAllWindows()
