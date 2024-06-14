import shutil
import tempfile
import ultralytics
from ultralytics import YOLO
import cv2
import pandas as pd
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import pytesseract  # Ajoutez cette ligne pour importer pytesseract
import streamlit as st
import cv2
from PIL import Image
import numpy as npe
import visualize
from visualize import draw_border
import os 
import matplotlib.pyplot as plt 
import glob 
import easyocr





# Remplacez ce chemin par le chemin correct de votre installation Tesseract
# Spécifiez explicitement le chemin vers l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Définir la variable d'environnement TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Page d'accueil
def page_accueil():
    # Ajouter le titre avec une icône et changer la couleur
    st.markdown(
        """
        <h1 style='color:#6495ED;'>
            <i class="fas fa-exclamation-triangle"></i> Welcome to BF plaques detector
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Pour que les icônes fonctionnent, ajoutez le lien vers le CDN de Font Awesome
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        """,
        unsafe_allow_html=True
    )

    # Section d'importance du système
    st.markdown(
    """
    <h3 style='background-color: #6495ED; color: white; padding: 10px;'>
        la nouvelle immatriculation
    </h3>
    """,
    unsafe_allow_html=True
    )

    st.write("La particularité de ce système par rapport à l’ancien système, "
             "a fait savoir le ministre, est que chaque catégorie de cycles (véhicules, motocycles, "
             "etc) a son fond de plaques. « Par exemple, le fond jaune est destiné aux véhicules et"
             " cycles à moteur des personnes privées, le fond blanc aux véhicules et cycles à moteur "
             "des collectivités publiques territoriales et celle des sociétés d’État et des établissements "
             "publics  autres qu’administratifs,  etc ». Le fond est suivi d’un numéro d’ordre, puis d’un groupe "
             "alphanumérique (lettre alphabet et un système de numérotation des véhicules, de gestion des "
             "couleurs des plaques et des types de véhicules) et d’un code de région, du régime douanier suspensif "
             "(cas particulier des IT et AT) également d’un code de sécurité.")

    # Tableau

    st.markdown(
    """
    <h3 style='background-color: #6495ED; color: white; padding: 10px;'>
        Caractéristiques du Système
    </h3>
    """,
    unsafe_allow_html=True
    )

    
    table_data = {
        "La couleur du fond de la plaque": ["Rouge ", "Blanc ", "Blanc ", "Jaune  ", "Vert ", "Bleu"],
        "La couleur de l’écriture sur la plaque ": ["RougeBlanc ", "Rouge ", "Bleue ", "noire  ", "orange ", "blanc"],
        "Types de personnes ou de structures qui l’utilisent": ["État, établissement public à caractère administratifs "
            , "Collectivité public territoriale ", "Société d’état et des établissement public autres qu’administratif "
            , "Usage privé  ", "Corps diplomatique consulaire et organismes internationaux ",
                                                                "Transport Public routier"]
    }
    st.table(table_data)

    # Section de symbolique des nouvelles plaques du Burkina Faso
 
    st.markdown(
    """
    <h3 style='background-color: #6495ED; color: white; padding: 10px;'>
        Symbolique des Nouvelles Plaques du Burkina Faso
    </h3>
    """,
    unsafe_allow_html=True
    )
    st.write("Les nouvelles plaques d'immatriculation du Burkina Faso ont été introduites...")

    # Image
    st.markdown(
    """
    <h3 style='background-color: #6495ED; color: white; padding: 10px;'>
        Nouvelles Plaques du Burkina Faso
    </h3>
    """,
    unsafe_allow_html=True
    )

    
 

    # Charger les images
    image1 = Image.open("models/th1.png")
    image2 = Image.open("models/th2.png")
    image3 = Image.open("models/th3.png")
    image4 = Image.open("models/th4.png")
    image5 = Image.open("models/th5.png")
    image6 = Image.open("models/th6.png")

    # Créer deux colonnes
    col1, col2 = st.columns(2)

    # Afficher les images dans les colonnes
    with col1:
        st.image(image1, caption="Nouvelles Plaques 1", use_column_width=True)
        st.image(image3, caption="Nouvelles Plaques 3", use_column_width=True)
        st.image(image5, caption="Nouvelles Plaques 5", use_column_width=True)
    

    with col2:
        st.image(image2, caption="Nouvelles Plaques 2", use_column_width=True)
        st.image(image4, caption="Nouvelles Plaques 4", use_column_width=True)
        st.image(image6, caption="Nouvelles Plaques 6", use_column_width=True)

    # Section d'importance du système
    st.markdown(
    """
    <h3 style='background-color: #6495ED; color: white; padding: 10px;'>
        Importance du Système de Détection de Plaques
    </h3>
    """,
    unsafe_allow_html=True
    )
   
    st.write("L'utilisation combinée de YOLO pour la détection et d'OCR pour la reconnaissance"
             " permet de créer un système de sécurité avancé qui peut détecter, identifier et "
             "suivre les véhicules en temps réel, contribuant ainsi à améliorer la sécurité dans "
             "diverses applications,"
             " comme la surveillance des parkings, le contrôle d'accès et la sécurité routière..")

   
def main():

    if __name__ == "__main__":
        main()



#******************************************************************************************
#******************************* start images****************************************************



import cv2
import numpy as np
from database import SessionLocal, Detection ,WebcamDetection ,VideoDetection
from datetime import datetime
from sqlalchemy import desc


# Définition de la fonction d'enregistrement des détections
def record_detection(image_path, concatenated_text):
    session = SessionLocal()
    date_time = datetime.now()
    new_detection = Detection(
        image_name=image_path,
        
        recognized_text=concatenated_text,
        
        date=date_time
    )
    session.add(new_detection)
    session.commit()
    session.close()

def detect_and_recognize_license_plates(image, plate_model):
    plate_results = plate_model(image)[0]
    threshold = 0.5
    results = {}

    for result in plate_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            license_plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            reader = easyocr.Reader(['en'])
            result = reader.readtext(license_plate_crop)
            concatenated_text = ' '.join([res[1] for res in result])
            
            if len(concatenated_text) >= 10:  # Vérifier la longueur du texte reconnu
                results = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': concatenated_text,
                        'bbox_score': score,
                    }
                }
                return results

    return None

def page_demo_image():
    st.markdown(
        """
        <h2 style='background-color: #6495ED; color: white; padding: 10px;'>
            Détection de véhicules et de plaques d'immatriculation
        </h2>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        temp_dir = tempfile.TemporaryDirectory()
        image_path = os.path.join(temp_dir.name, "uploaded_image.jpg")

        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        original_image = Image.open(image_path)
        vehicle_model_path = 'yolov8n.pt'
        vehicle_model = YOLO(vehicle_model_path)
        plate_model_path = './models/last.pt'
        plate_model = YOLO(plate_model_path)

        image = cv2.imread(image_path)
        vehicle_results = vehicle_model(image)[0]
        plate_results = plate_model(image)[0]

        threshold = 0.5
        detected_vehicles = 0
        detected_plates = 0

        for result in vehicle_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(image, vehicle_results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                detected_vehicles += 1

        for result in plate_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(image, plate_results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                detected_plates += 1  # Incrémentez le compteur

        results = detect_and_recognize_license_plates(image, plate_model)

        st.markdown(
            """
            <h4 style='background-color: #6495ED; color: white; padding: 10px;'>
                Détection des plaques 
            </h4>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Image d'origine", use_column_width=True)
            st.write(f"Nombre de véhicules détectés : {detected_vehicles}")
        with col2:
            st.image(image, caption="Image annotée", use_column_width=True)
            st.write(f"Nombre de plaques d'immatriculation détectées : {detected_plates}")
        temp_dir.cleanup()

        st.markdown(
            """
            <h4 style='background-color: #6495ED; color: white; padding: 10px;'>
                Reconnaissance de Caractères sur les plaques et performance  
            </h4>
            """,
            unsafe_allow_html=True
        )

        if results and 'license_plate' in results:
            record_detection(image_path, results['license_plate']['text'])
            st.write('Texte de la plaque d\'immatriculation :', results['license_plate']['text'])

    session = SessionLocal()
    st.title("Tableau des Détections")
    detections = session.query(Detection).order_by(desc(Detection.date)).limit(4).all()
    data = [
        {
            "Numero": detection.id,
            "Nom de l'image": detection.image_name,
            "Texte reconnu": detection.recognized_text,
            "Date": detection.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        for detection in detections
    ]

    df = pd.DataFrame(data)
    st.dataframe(df)
    session.close()



#******************************************************************************************
#*******************************start video ****************************************************



def insert_video_detection(image_name, recognized_text):
    session = SessionLocal()
    new_video_detection = VideoDetection(
        image_name=image_name,
        recognized_text=recognized_text,
        date=datetime.utcnow()
    )
    session.add(new_video_detection)
    session.commit()
    session.close()

def detect_and_recognize_license_plates(image, plate_model):
    plate_results = plate_model(image)[0]
    threshold = 0.5
    results = {}

    for result in plate_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            license_plate_crop = image[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(license_plate_crop)
            concatenated_text = ' '.join([res[1] for res in result])
            if len(concatenated_text) >= 8:  # Vérifier la longueur du texte reconnu
                results = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': concatenated_text,
                        'bbox_score': score,
                    }
                }
                return results  # Quitter la fonction dès qu'une plaque est détectée

    return None  # Aucune plaque détectée

def page_demo_video():
    st.markdown(
        """
        <h2 style='background-color: #6495ED; color: white; padding: 10px;'>
            Détection de véhicules et de plaques d'immatriculation à partir de vidéos
        </h2>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video("uploaded_video.mp4")

        cap = cv2.VideoCapture("uploaded_video.mp4")
        plate_model_path = './models/last.pt'
        plate_model = YOLO(plate_model_path)
        
        video_placeholder = st.empty()
        st.markdown(
            """
            <h4 style='background-color: #6495ED; color: white; padding: 10px;'>
                Reconnaissance de Caractères sur la plaque d'immatriculation
            </h4>
            """,
            unsafe_allow_html=True
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur lors de la lecture de la vidéo.")
                break

            results = detect_and_recognize_license_plates(frame, plate_model)

            if results is not None and len(results['license_plate']['text']) >= 8:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(frame[int(results['license_plate']['bbox'][1]):int(results['license_plate']['bbox'][3]),
                                  int(results['license_plate']['bbox'][0]):int(results['license_plate']['bbox'][2])],
                             caption="Capture de la plaque détectée",
                             use_column_width=True)
                with col2:
                    st.write('Texte de la plaque d\'immatriculation :', results['license_plate']['text'])

                insert_video_detection("video_capture.jpg", results['license_plate']['text'])

                x1, y1, x2, y2 = results['license_plate']['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame, results['license_plate']['text'], (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.1)

        cap.release()

        session = SessionLocal()
        st.title("Tableau des Détections")
        detections = session.query(VideoDetection).order_by(desc(VideoDetection.date)).limit(5).all()
        data = [
            {
                "Numero": detection.id,
                "Nom de l'image": detection.image_name,
                "Texte reconnu": detection.recognized_text,
                "Date": detection.date.strftime('%Y-%m-%d %H:%M:%S')
            }
            for detection in detections
        ]
        df = pd.DataFrame(data)
        st.dataframe(df)
        session.close()

#******************************************************************************************
#*******************************start webcam ****************************************************






# Définition de la fonction d'enregistrement des détections
def insert_webcam_detection(image_name, recognized_text):
    session = SessionLocal()
    new_webcam_detection = WebcamDetection(
        image_name=image_name,
        recognized_text=recognized_text,
        date=datetime.utcnow()
    )
    session.add(new_webcam_detection)
    session.commit()
    session.close()

# Définition de la fonction de détection et de reconnaissance des plaques d'immatriculation
def detect_and_recognize_license_plates(image, plate_model):
    plate_results = plate_model(image)[0]
    threshold = 0.5
    results = {}

    for result in plate_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            license_plate_crop = image[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                 cv2.THRESH_BINARY_INV)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(license_plate_crop)
            concatenated_text = ' '.join([res[1] for res in result])
            if len(concatenated_text) >= 8:  # Vérifier la longueur du texte reconnu
                results = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': concatenated_text,
                        'bbox_score': score,
                    }
                }
                return results  # Quitter la fonction dès qu'une plaque est détectée

    return None  # Aucune plaque détectée

# Définir la page
def page_demo_webcam():
    st.markdown(
        """
        <h2 style='background-color: #6495ED; color: white; padding: 10px;'>
            Détection de véhicules et de plaques d'immatriculation en temps réel avec une webcam
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Variables de contrôle pour le démarrage et l'arrêt de la webcam
    if 'run' not in st.session_state:
        st.session_state.run = False

    def start():
        st.session_state.run = True

    def stop():
        st.session_state.run = False

    # Boutons Start et Stop
    st.button("Start", on_click=start)
    st.button("Stop", on_click=stop)

    # Initialiser la capture vidéo à partir de la webcam
    cap = cv2.VideoCapture(0)
    plate_model_path = './models/last.pt'
    plate_model = YOLO(plate_model_path)

    # Placeholder pour la vidéo
    video_placeholder = st.empty()
    st.markdown(
        """
        <h4 style='background-color: #6495ED; color: white; padding: 10px;'>
            Reconnaissance de Caractères sur la plaque d'immatriculation
        </h4>
        """,
        unsafe_allow_html=True
    )

    # Boucle pour capturer et traiter chaque image de la webcam
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture de la vidéo.")
            break

        # Détecter et reconnaître les plaques d'immatriculation
        results = detect_and_recognize_license_plates(frame, plate_model)

        # Si une plaque est détectée, afficher la capture et le texte reconnu dans deux colonnes
        if results is not None and len(results['license_plate']['text']) >= 8:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(frame[int(results['license_plate']['bbox'][1]):int(results['license_plate']['bbox'][3]),
                              int(results['license_plate']['bbox'][0]):int(results['license_plate']['bbox'][2])],
                         caption="Capture de la plaque détectée",
                         use_column_width=True)
            
            with col2:
                st.write('Texte de la plaque d\'immatriculation :', results['license_plate']['text'])
            
            insert_webcam_detection("webcam_capture.jpg", results['license_plate']['text'])

            # Annoter l'image
            x1, y1, x2, y2 = results['license_plate']['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(frame, results['license_plate']['text'], (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

        # Afficher la vidéo en direct avec les annotations
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Attendre un petit moment avant de capturer la prochaine image
        time.sleep(0.1)

    # Fermer la capture vidéo et libérer les ressources
    cap.release()

    # Afficher les détections enregistrées dans la base de données
    session = SessionLocal()
    st.title("Tableau des Détections")
    detectionscam = session.query(WebcamDetection).order_by(desc(WebcamDetection.date)).limit(5).all()
    data = [
        {
            "Numero": detection.id,
            "Nom de l'image": detection.image_name,
            "Texte reconnu": detection.recognized_text,
            "Date": detection.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        for detection in detectionscam
    ]
    df = pd.DataFrame(data)
    st.dataframe(df)
    session.close()












# Page pour afficher les résultats enregistrés
def page_show_detections():
    st.title("Historiques des Détections webcam")

    # Afficher les détections de la table WebcamDetection
    session = SessionLocal()
    webcam_detections = session.query(WebcamDetection).all()
    session.close()

    data_webcam = [
        {
            "Numero": webcam_detection.id,
            "Nom de l'image": webcam_detection.image_name,
            "Texte reconnu": webcam_detection.recognized_text,
            "Date": webcam_detection.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        for webcam_detection in webcam_detections
    ]
    df_webcam = pd.DataFrame(data_webcam)
    st.dataframe(df_webcam)

    st.title("Historiques des Détections image ")

    # Afficher les détections de la table Detection
    session = SessionLocal()
    detections = session.query(Detection).all()
    session.close()

    data_detection = [
        {
            "Numero": detection.id,
            "Nom de l'image": detection.image_name,
            "Texte reconnu": detection.recognized_text,
            "Date": detection.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        for detection in detections
    ]
    df_detection = pd.DataFrame(data_detection)
    st.dataframe(df_detection)

    st.title("Historiques des Détections video ")
 # Afficher les détections de la table video
    session = SessionLocal()
    video_detections = session.query(VideoDetection).all()
    session.close()

    data_video = [
        {
            "Numero":  video_detection.id,
            "Nom de l'image":  video_detection.image_name,
            "Texte reconnu":  video_detection.recognized_text,
            "Date":  video_detection.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        for  video_detection in  video_detections
    ]
    df_video = pd.DataFrame(data_video)
    st.dataframe(df_video)



#******************************************************************************************
#*******************************side bar ****************************************************
# Lancement de l'application
if __name__ == "__main__":
    # Charger l'image

    # Charger l'image
    logo = Image.open("models/logo.png")

    # Afficher l'image dans Streamlit
    st.sidebar.image(logo, caption='', use_column_width=True)
       
    st.sidebar.title("Menu")
   
    menu_options = ["Accueil", "Démo Image", "Démo Vidéo", "Démo Webcam", "Historiques"]
    selected_menu = st.sidebar.radio("Sélectionnez une option :", menu_options)

    if selected_menu == "Accueil":
        page_accueil()
    if selected_menu == "Démo Image":
        page_demo_image()
    if selected_menu == "Démo Vidéo":
        page_demo_video()
    if selected_menu == "Démo Webcam":
        page_demo_webcam()
    elif selected_menu == "Historiques":
        page_show_detections()




