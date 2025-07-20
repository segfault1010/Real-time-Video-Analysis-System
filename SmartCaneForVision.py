import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model


# Initialize models and configurations
def initialize_models():
    # Object detection model (MobileNet SSD)
    obj_net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'mobilenet_iter_73000.caffemodel'
    )
    obj_classes = ["background", "person", "bicycle", "car", "motorcycle",
                   "airplane", "bus", "train", "truck", "boat", "traffic light",
                   "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                   "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                   "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                   "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                   "kite", "baseball bat", "baseball glove", "skateboard",
                   "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple",
                   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                   "donut", "cake", "chair", "couch", "potted plant", "bed",
                   "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                   "keyboard", "cell phone", "microwave", "oven", "toaster",
                   "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]

    # Emotion recognition model
    emotion_model = load_model('emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Known face database (add your own images)
    known_face_encodings = []
    known_face_names = []

    return obj_net, obj_classes, emotion_model, emotion_labels, known_face_encodings, known_face_names


# Process frame for object detection
def detect_objects(frame, net, classes):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{classes[idx]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Process frame for face and emotion recognition
def recognize_faces_and_emotions(frame, emotion_model, emotion_labels, known_encodings, known_names):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Face recognition
        name = "Unknown"
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

        # Emotion recognition
        face_roi = frame[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
        emotion_prediction = emotion_model.predict(reshaped_face)
        emotion_idx = np.argmax(emotion_prediction)
        emotion = emotion_labels[emotion_idx]

        # Draw results
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, f"{name}: {emotion}", (left, top - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


# Main processing function
def main():
    # Initialize models
    obj_net, obj_classes, emotion_model, emotion_labels, known_encodings, known_names = initialize_models()

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        detect_objects(frame, obj_net, obj_classes)

        # Face and emotion recognition
        recognize_faces_and_emotions(frame, emotion_model, emotion_labels, known_encodings, known_names)

        # Display results
        cv2.imshow('Live Analysis', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()