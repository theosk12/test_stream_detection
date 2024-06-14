

import streamlit as st
import base64
import main  # Importez le module main.py
import visualize  # Importez le module visualize.py

# Définition d'une fonction pour la détection de plaques d'immatriculation
def page_detection():
    st.title("Détection de plaques d'immatriculation")
    uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4"])

    if uploaded_video is not None:
        # Exécutez la détection depuis main.py et obtenez le fichier CSV
        results_csv = main.run_detection(uploaded_video)

        # Affichez le nombre de plaques détectées
        st.write(f"Nombre total de plaques d'immatriculation détectées : {len(results_csv)}")

        # Affichez un lien pour télécharger le fichier CSV
        st.markdown(get_csv_download_link(results_csv), unsafe_allow_html=True)

# Définition d'une fonction pour la visualisation de la vidéo avec les boîtes de détection
def page_visualization():
    st.title("Visualisation de la vidéo avec les boîtes de détection")
    uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4"])

    if uploaded_video is not None:
        # Exécutez la visualisation depuis visualize.py
        visualize.run_visualization(uploaded_video)

# Fonction pour générer un lien de téléchargement pour un fichier CSV
def get_csv_download_link(csv_data):
    csv_file = csv_data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Télécharger le fichier CSV</a>'
    return href

# Créez une barre de navigation pour basculer entre les pages
app_mode = st.sidebar.selectbox("Sélectionnez une page", ["Détection de plaques", "Visualisation"])

# En fonction de la page sélectionnée, affichez la page correspondante
if app_mode == "Détection de plaques":
    page_detection()
elif app_mode == "Visualisation":
    page_visualization()
