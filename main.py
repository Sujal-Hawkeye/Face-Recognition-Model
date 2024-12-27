import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing training images
path = "D:\\village\\Face-Recognition-model\\Training_images"
images = []
classNames = []

# Specify the directory to store the CSV file
attendance_directory = "D:\\village\\Face-Recognition-model\\Attendance_Records\\"  # Modify this to your preferred directory

# Create the directory if it doesn't exist
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

# Path for the attendance CSV file
attendance_file_path = os.path.join(attendance_directory, "Attendance.csv")

# Initialize the attendance dictionary
attendance_dict = {}

# Ensure the path does not contain leading/trailing spaces
myList = os.listdir(path)
print("Image list:", myList)

# Load images and their corresponding class names
for cl in myList:
    curImg = cv2.imread(f'{path}\\{cl}')
    if curImg is None:
        print(f"Failed to load image {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Initialize the attendance dictionary with 'Absent' status for each class name
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

# Create the CSV file with headers (if it doesn't exist)
def createCSV():
    if not os.path.isfile(attendance_file_path):
        with open(attendance_file_path, 'w') as f:
            f.write('Name,Time,Status\n')

# Write attendance to CSV file
def updateAttendanceCSV(attendance_dict):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    with open(attendance_file_path, 'a') as f:
        for name, status in attendance_dict.items():
            f.write(f'{name},{dtString},{status}\n')

# Find encodings of training images
encodeListKnown = findEncodings(images)
print('Encodings Complete')

# Create the CSV file
createCSV()

# Start video capture
cap = cv2.VideoCapture(0)

# Track which people have been marked as 'Present'
present_set = set()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    # Resize the frame to 1/4 size for faster processing
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings for the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop through each face in the current frame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if name not in present_set:  # Check if already marked as 'Present'
                print(f"Recognized {name}")  # Only print once when the person is recognized
                attendance_dict[name] = 'Present'  # Mark the person as present
                present_set.add(name)  # Add to the set to avoid further prints

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame with names marked as present
    cv2.imshow('Webcam', img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Mark all people who were not detected as 'Absent'
for name in classNames:
    if name.upper() not in present_set:
        attendance_dict[name.upper()] = 'Absent'

# Update the attendance CSV with present and absent students
updateAttendanceCSV(attendance_dict)

cap.release()
cv2.destroyAllWindows()
