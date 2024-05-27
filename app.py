import streamlit as st
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# SE CORRE CON: python -m streamlit run app.py


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        hand_detected = False
        if self.results.multi_hand_landmarks:
            hand_detected = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return hand_detected, img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

# Cargar el modelo
model = tf.keras.models.load_model('hand_gesture_model_0_to_5.h5')

# Inicializar el detector de manos
detector = handDetector(detectionCon=0.75)

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = handDetector(detectionCon=0.75)
        self.model = tf.keras.models.load_model('hand_gesture_model_0_to_5.h5')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detectar manos en la imagen
        hand_detected, img = self.detector.findHands(img)

        if hand_detected:
            # Preprocesar la imagen para la inferencia
            img_resized = cv2.resize(img, (64, 64))  # Asumiendo que el modelo espera imágenes de 64x64
            img_resized = img_resized / 255.0
            img_expanded = np.expand_dims(img_resized, axis=0)

            # Realizar la predicción
            prediction = self.model.predict(img_expanded)
            predicted_number = np.argmax(prediction)

            # Escribir el número predicho en la imagen
            cv2.putText(img, f'Predicted: {predicted_number}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

            # Enviar el número detectado a la UI en HTML
            st.write(f"<script>parent.postMessage({predicted_number}, '*');</script>", unsafe_allow_html=True)
        return img

st.title('Hand Gesture Recognition in Real-Time')
webrtc_streamer(key="hand-gesture", video_transformer_factory=HandGestureTransformer)
