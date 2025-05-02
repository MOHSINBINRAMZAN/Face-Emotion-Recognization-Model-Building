

# 😊 Face Emotion Recognition Model

This project focuses on building a **Face Emotion Recognition** system using **Computer Vision** and **Deep Learning** techniques. The model is trained to detect facial expressions and classify emotions such as *Happy, Sad, Angry, Fear, Surprise, Neutral,* and more.

## 🚀 Features

* 🤖 **Emotion Classification**: Detects emotions in real-time from facial images or video streams.
* 🎥 **Webcam Integration**: Run the model live using your device's camera.
* 📊 **Custom or Pre-trained Models**: Option to train from scratch or use transfer learning (e.g., VGG16, ResNet).
* 🖼️ **Face Detection**: Uses Haar Cascades or MTCNN/Dlib for face localization.
* 🧠 **CNN Architecture**: Deep Convolutional Neural Networks (CNN) for high emotion classification accuracy.
* 💾 **FER2013 Dataset Support**: Preprocessed dataset used for model training and testing.

## 🛠️ Tech Stack

* **Language**: Python
* **Libraries**: TensorFlow / Keras / PyTorch, OpenCV, NumPy, Matplotlib
* **Dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) or custom labeled datasets


## 💡 How It Works

1. **Face Detection**: Extract face region from image/video.
2. **Preprocessing**: Resize, normalize, and grayscale the input.
3. **Emotion Prediction**: Run face through trained CNN to get predicted emotion.
4. **Visualization**: Display real-time emotion label with bounding box.

## 🧪 Sample Prediction Code

```python
import cv2
from keras.models import load_model

model = load_model('models/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read from webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        emotion = emotion_labels[model.predict(face).argmax()]
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/face-emotion-recognition.git
cd face-emotion-recognition
pip install -r requirements.txt
python emotion_model.py         # Train model
python predict.py               # Predict emotions via webcam
```

## 📈 Accuracy

Model achieves around **65-75% accuracy** on the FER2013 validation set, depending on the architecture and preprocessing steps used.

## 📌 Future Improvements

* Add mobile support via TensorFlow Lite
* Train on custom datasets for better domain-specific accuracy
* Use attention-based or transformer models for improved results
* Integrate into virtual meeting platforms or customer service bots

## 📄 License

This project is licensed under the [MIT License](LICENSE).

