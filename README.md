# Face Recognition Attendance Model

This is a Python-based project for automating attendance systems using face recognition. Designed to streamline real-time attendance tracking, it employs machine learning and computer vision techniques.

## Key Features

- **Face Detection & Alignment**: Accurate face detection using Haar cascades.
- **Feature Extraction**: Leverages pre-trained models like FaceNet for embedding generation.
- **Automated Attendance**: Matches recognized faces to a database for seamless tracking.
- **Real-Time Processing**: Supports webcam or video feed for dynamic attendance.
- **Exportable Reports**: Attendance logs are generated in CSV format.

## Requirements

- Python 3.7+
- Libraries:
  - OpenCV
  - TensorFlow/Keras
  - Pandas
  - Dlib
- (Optional) CUDA and cuDNN for GPU acceleration

## Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
  
Running the Attendance System:

Simply run the Python script:

 ```bash
face_recognition_attendance.py
   ```
How it Works:
1. When you start the script, it will open a webcam window.
2. The program will look for faces in the webcam feed and match them with the pre-trained faces stored in Training_images.
3. If a face is recognized, it will mark the person as "Present" in the CSV file.
4. If a face is not recognized, it will mark the individual as "Absent".
5. Press the 'q' key to exit the program.


Attendance CSV:

The attendance data will be saved in a CSV file (Attendance.csv) located in a user-specified directory (by default, D:\\Attendance_Records\\).
The CSV will store the name, time, and status ("Present"/"Absent") for each individual.

## Advanced Options

- Integrate with SQL/NoSQL databases for log storage.
- Add notifications via email or SMS.
- Use custom models for specific requirements.


## Notes for Developers

- Modify `train_model.py` or `real_time_attendance.py` for project-specific needs.
- Test performance with a variety of datasets to ensure accuracy.


