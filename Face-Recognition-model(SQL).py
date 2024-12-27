import cv2
import numpy as np
import face_recognition
import os
import mysql.connector
from datetime import datetime

path = "D:\\village\\Face-Recognition-model\\Training_images"
images = []
classNames = []

myList = os.listdir(path)
print("Image list:", myList)

for cl in myList:
    curImg = cv2.imread(f'{path}\\{cl}')
    if curImg is None:
        print(f"Failed to load image {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class names:", classNames)

attendance_dict = {name.upper(): 'Absent' for name in classNames}

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

def connectToDatabase():
    conn = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "",
        database = "SujalDB"
    )
    return conn

def initializeDatabase():
    conn = connectToDatabase()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                      (Name VARCHAR(255), Time VARCHAR(255), Status VARCHAR(255))''')
    conn.commit()
    cursor.close()
    conn.close()

def updateAttendanceDatabase(name, status):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S %d-%m-%Y')

    conn = connectToDatabase()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (Name, Time, Status) VALUES (%s, %s, %s)",
                   (name, dtString, status))
    
    conn.commit()
    cursor.close()
    conn.close()

# def markAllAbsent():
#     now = datetime.now()
#     dtString = now.strftime('%H:%M:%S %d-%m-%Y')
#     conn = connectToDatabase()
#     cursor = conn.cursor()
#     for name in classNames:
#         cursor.execute("INSERT INTO attendance (Name, Time, Status) VALUES (%s, %s, %s)",
#                        (name.upper(), dtString, 'Absent'))
#     conn.commit()
#     cursor.close()
#     conn.close()


encodeListKnown = findEncodings(images)
print('Encodings Complete')


initializeDatabase()

cap = cv2.VideoCapture(0)

try:
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
                if attendance_dict[name] == 'Absent':  
                    attendance_dict[name] = 'Present'
                    updateAttendanceDatabase(name, 'Present')
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
