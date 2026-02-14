import cv2
import csv
import os
import easyocr
import argparse
from ultralytics import YOLO
from cvzone.Utils import cornerRect, putTextRect

parser = argparse.ArgumentParser(description="Reconocimiento de matrículas")
parser.add_argument(
    "--video",
    help="Ruta del video o URL (RTSP, HTTP...)",
    required=True
)
coco_model = YOLO('yolo11x.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
reader = easyocr.Reader(['en'], gpu=False)
csv_filename = 'matriculas.csv'
#Si el archivo CSV no existe, se crea
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_number', 'car_id', 'license_plate'])

#Diccionario para mapear las clases de vehículos a sus nombres
vehicles = {2: "Coche", 3: "Moto", 5: "Autobus", 7: "Camion"} 

frame_number = 0

bytetrack_tracker = "bytetrack.yaml"

vehicle_plates = {}

cap = cv2.VideoCapture('./sample4.mp4')

ret,frame= cap.read()
# Si no se pudo leer el video, se muestra un mensaje de error y se cierra el programa
if not ret:
    print('Error al leer el video')
    cap.release()
    cv2.destroyAllWindows()
    exit()


#Seleccionamos la region de interes para no analizar todo el video
roi = cv2.selectROI("Selecciona la region de interes",frame,fromCenter=False,showCrosshair=True)
cv2.destroyWindow("Selecciona la region de interes")

x_roi, y_roi, w_roi, h_roi = roi


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame_number += 1
    #Extraemos la region de interes del frame
    roi_frame = frame[y_roi:y_roi+h_roi,x_roi: x_roi+w_roi].copy()
    # Le pasamos la region de interes al modelo de deteccion y seguimiento
    results = coco_model.track(roi_frame, persist=True, tracker=bytetrack_tracker, classes=list(vehicles.keys()), iou=0.5, agnostic_nms=True)

    vehicle_tracks = {}

    # Verificamos si se han detectado vehículos y si tienen un ID de seguimiento asignado
    if results[0].boxes.id is not None:
              for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        track_id = int(track_id)
                        class_id = int(class_id)

                        x1 += x_roi
                        x2 += x_roi
                        y1 += y_roi
                        y2 += y_roi

                        vehicle_tracks[track_id] = (x1, y1, x2, y2)
    # Para cada vehiculo detectado, se busca una matricula dentro de su area
    license_plates = license_plate_detector(roi_frame)[0]

    # Si se han detectado matrículas, se verifica si alguna de ellas está dentro del área de algún vehículo detectado
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        x1 += x_roi
        x2 += x_roi
        y1 += y_roi
        y2 += y_roi
        # Se verifica si la matrícula detectada está dentro del área de algún vehículo detectado
        for track_id, (xcar1, ycar1, xcar2, ycar2) in vehicle_tracks.items():
              # Si la matrícula está dentro del área del vehículo, se recorta la matrícula, se procesa con pytesseract y se guarda la información en el CSV
              if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                   
                    plate = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # Pasar a gris
                    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                    # Filtro para mantener bordes
                    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

                    # Binarización adaptativa (MUY importante)
                    plate_gray = cv2.adaptiveThreshold(
                        plate_gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        31,
                        2
                    )
                    license_text = ""
                    # Utilizamos EasyOCR para leer el texto de la matrícula
                    results = reader.readtext(plate_gray)

                    for (bbox, text, conf) in results:
                        # Solo consideramos resultados con una confianza mayor a 0.5
                        if conf > 0.5:
                            text = text.upper().replace(" ", "")
                            license_text = text
               

                    if track_id not in vehicle_plates or len(license_text) > len(vehicle_plates[track_id]):
                        vehicle_plates[track_id] = license_text


                    with open(csv_filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([frame_number,track_id,vehicle_plates[track_id]])

                        # Dibujamos un rectángulo alrededor del vehículo y mostramos la matrícula detectada
                        cornerRect(frame, (int(xcar1), int(ycar1), int(xcar2 - xcar1), int(ycar2 - ycar1)), l=10, rt=2, colorR=(255, 0, 0))

                        putTextRect(frame, f'Car {track_id}', (int(xcar1), int(ycar1) - 10), scale=0.8, thickness=2, colorR=(255, 0, 0), colorB=(255, 255, 255))

                        putTextRect(frame, vehicle_plates[track_id], (int(x1), int(y1) - 10), scale=3.3, thickness=2, colorR=(0, 0, 0), colorB=(255, 255, 255), border=3)

    cv2.imshow('Vehiculo y Matricula', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()