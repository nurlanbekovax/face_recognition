import cv2
import numpy as np
import os

# Путь к датасету
dataset_path = "dataset"
known_faces = []
known_names = []

# Загрузка и обучение на изображениях из датасета
print("Training on dataset...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Процесс обучения
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # Загрузка изображения
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))  # Приведение лиц к одному размеру
            # Считаем гистограмму
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            known_faces.append(hist)
            known_names.append(person_name)

print("Training complete!")

# Инициализация камеры
video_capture = cv2.VideoCapture(0)

# Порог для минимальной разницы (можно оставить как есть)
threshold = 0.8  # Порог для сравнения гистограмм (чем ближе к 1, тем более похожи лица)

print("Starting face recognition...")
while True:
    # Считывание кадра с камеры
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))

        # Считаем гистограмму
        hist_frame = cv2.calcHist([face_resized], [0], None, [256], [0, 256])

        name = "Unknown"
        max_corr = 0

        # Сравнение гистограмм
        for known_face, person_name in zip(known_faces, known_names):
            corr = cv2.compareHist(hist_frame, known_face, cv2.HISTCMP_CORREL)  # Корреляция гистограмм
            if corr > max_corr and corr > threshold:
                max_corr = corr
                name = person_name

        # Отобразить имя на кадре
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Отображение результата в окне
    cv2.imshow("Face Recognition", frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
