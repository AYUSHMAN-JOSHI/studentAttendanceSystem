import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json
import time
import random

# Function to register a person and generate a JSON record
def register_person(person_name):
    data = {
        'name': person_name,
        'ID': random.randint(2016000, 2017000)
    }
    return data


# Path to the folder containing training images
# C:\Users\ayush\OneDrive - Graphic Era University\Desktop\FOLDERS\MY FOLDER\COLLEGE\SUBJECTS\IMAGE PROCESSING & COMPUTER VISION\PROJECT\images
train_path = r'C:\\Users\\ayush\\OneDrive - Graphic Era University\Desktop\\FOLDERS\\MY FOLDER\\COLLEGE\SUBJECTS\\IMAGE PROCESSING & COMPUTER VISION\\PROJECT\\images'

# Get the list of people from the training folder
people = os.listdir(train_path)

# Prepare empty lists to store the training data and labels
X_train = []
y_train = []

# Load and preprocess training images
for person_id, person in enumerate(people):
    person_path = os.path.join(train_path, person)
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (128, 128))  # Resize to a fixed size
        X_train.append(image)
        y_train.append(person_id)

# Create a dictionary to map label indices to person names
label_to_person = {i: person for i, person in enumerate(people)}

# Convert the training data and labels to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize the pixel values to the range of [0, 1]
X_train = X_train.astype('float32') / 255.0

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))  # Updated input shape
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(people), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture from the default camera
video_capture = cv2.VideoCapture(0)

# Variable for tracking time
last_detection_time = time.time()

# Initialize the person_detected variable
person_detected = False

# List to hold all student records
student_records = []

while True:
    # Read the current frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Reset the person_detected variable at the start of each loop iteration
    person_detected = False

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face image
        face = cv2.resize(face, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face, axis=0)
        face = face.astype('float32') / 255.0

        # Make prediction on the face
        prediction = model.predict(face)
        label = np.argmax(prediction)

        # Update the person_detected variable if a person is detected
        person_detected = True

        # last_detection_time = time.time()
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, people[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # If a person is detected for 3-5 seconds, register them
    if person_detected:
        if time.time() - last_detection_time >= 5:
            student_name = label_to_person[label]
            student_record = register_person(student_name)
            
            # Check if the record already exists in student_records
            if student_record["name"] not in student_records:
                student_records.append(student_record)

            last_detection_time = time.time()
        

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save all student records to a JSON file
json_path = 'student_records.json'
with open(json_path, 'w') as json_file:
    json.dump(student_records, json_file)

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
