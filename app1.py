import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image, ImageDraw, ImageOps

# Clase para detectar manos
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

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

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

    def normalize_hand(self, img, lmList):
        if not lmList:
            return None
        x_min = min([lm[1] for lm in lmList])
        y_min = min([lm[2] for lm in lmList])
        x_max = max([lm[1] for lm in lmList])
        y_max = max([lm[2] for lm in lmList])
        hand_img = img[y_min:y_max, x_min:x_max]
        standard_size = (200, 200)
        normalized_hand_img = cv2.resize(hand_img, standard_size, interpolation=cv2.INTER_AREA)
        return normalized_hand_img

# Cargar modelo y escalador
model = load_model('hand_gesture_model_0_to_5.h5')
scaler = joblib.load('scaler.pkl')

# Inicializar detector de manos
detector = handDetector(detectionCon=0.75)

# Configurar Streamlit
st.title('Sistema de Pedido de Comida por Gestos de Mano')
st.text('Muestra un número con tu mano (0-5) para seleccionar una opción del menú.')

# Opciones del menú
menu_options = {
    0: 'Hamburguesa',
    1: 'Pizza',
    2: 'Ensalada',
    3: 'Pasta',
    4: 'Tacos',
    5: 'Sushi'
}

menu_images = {
    0: './images/hamburguesa.jpg',
    1: './images/pizza.jpg',
    2: './images/ensalada.jpg',
    3: './images/pasta.jpg',
    4: './images/tacos.jpg',
    5: './images/sushi.jpg'
}

menu_options_final = {
    'Hamburguesa': './images/hamburguesa.jpg',
    'Pizza': './images/pizza.jpg',
    'Ensalada': './images/ensalada.jpg',
    'Pasta': './images/pasta.jpg',
    'Tacos': './images/tacos.jpg',
    'Sushi': './images/sushi.jpg'
}

selected_items = []

# Función para mostrar el círculo de progreso en una imagen separada
def draw_progress_circle(base_img, progress):
    circle_img = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(circle_img)
    angle = 360 * progress
    draw.arc((10, 10, base_img.size[0] - 10, base_img.size[1] - 10), start=0, end=angle, fill=(255, 0, 0, 255), width=10)
    combined = Image.alpha_composite(base_img.convert("RGBA"), circle_img)
    return combined

# Mostrar el menú en la barra lateral con imágenes más pequeñas en dos columnas
st.sidebar.title("Menú de Opciones")
col1, col2 = st.sidebar.columns(2)
menu_placeholders = {}
for idx, option in menu_options.items():
    if idx % 2 == 0:
        menu_placeholders[idx] = col1.empty()
    else:
        menu_placeholders[idx] = col2.empty()

for idx, option in menu_options.items():
    menu_placeholders[idx].image(menu_images[idx], caption=f"{idx}. {option}", use_column_width=True)

# Función para realizar inferencia en tiempo real
def real_time_inference():
    global start_time, selected_option
    pTime = 0
    selected_option = -1
    start_time = None
    stframe = st.empty()
    cap = cv2.VideoCapture(1)  # Seleccionar cámara 0
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30

    while True:
        success, img = cap.read()
        if not success:
            continue
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        normalized_hand = detector.normalize_hand(img, lmList)

        if normalized_hand is not None and len(lmList) == 21:
            lmArray = np.array([coord for lm in lmList for coord in lm[1:]]).reshape(1, -1)
            lmArray = scaler.transform(lmArray)

            prediction = model.predict(lmArray)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if selected_option == class_id:
                elapsed_time = time.time() - start_time
                progress = elapsed_time / 5.0
                if progress < 1.0:
                    base_img = Image.open(menu_images[class_id])
                    img_with_circle = draw_progress_circle(base_img, progress)
                    menu_placeholders[class_id].image(img_with_circle, caption=f"{class_id}. {menu_options[class_id]}", use_column_width=True)
                else:
                    selected_items.append(menu_options[class_id])
                    st.write(f'Has seleccionado: {menu_options[class_id]}')
                    if len(selected_items) >= 3:
                        st.write("Menú final seleccionado:")
                        for item in selected_items:
                            st.write(item)
                            st.image(menu_options_final[item])
                        break
                    for idx in menu_options.keys():
                        menu_placeholders[idx].image(menu_images[idx], caption=f"{idx}. {menu_options[idx]}", use_column_width=True)
                    time.sleep(2)  # Pausa para evitar selecciones consecutivas rápidas
                    selected_option = -1  # Reiniciar selección
            else:
                selected_option = class_id
                start_time = time.time()
                # Borrar progreso en todas las imágenes
                for idx in menu_options.keys():
                    menu_placeholders[idx].image(menu_images[idx], caption=f"{idx}. {menu_options[idx]}", use_column_width=True)

        else:
            selected_option = -1  # Reiniciar selección si no se detecta la mano

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        stframe.image(img, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar inferencia en tiempo real
if st.button('Iniciar detección'):
    real_time_inference()
