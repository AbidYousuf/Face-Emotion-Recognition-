from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the face classifier for detecting faces
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained MobileNetV2 model
model = load_model('modelA.keras')

# Define emotion labels (adjust these labels to match your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for emotion detection
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray_img, 1.1, 4)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Use the detected face ROI for further detection if needed
        facess = face_classifier.detectMultiScale(roi_gray)

        if len(facess) == 0:
            return jsonify({'error': 'No face detected within ROI'})

        # Crop the detected face region
        for ex, ey, ew, eh in facess:
            face_roi = roi_color[ey:ey + eh, ex:ex + ew]

        # Resize the face ROI to the input size expected by the model
        final_img = cv2.resize(face_roi, (224, 224))
        final_img = final_img / 255.0  # Normalize the image
        final_img = np.expand_dims(final_img, axis=0)

        # Make predictions
        prediction = model.predict(final_img)
        predicted_index = np.argmax(prediction)  # Get the index of the highest probability
        emotion = emotion_labels[predicted_index]  # Map index to emotion label

        # Return the prediction result
        return jsonify({'emotion': emotion})


if __name__ == "__main__":
    app.run(debug=True)
