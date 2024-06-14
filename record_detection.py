# record_detection.py
from database import SessionLocal, Detection
from datetime import datetime

def record_detection(image_path, bbox_score, concatenated_text, text_score):
    session = SessionLocal()
    date_time = datetime.now()
    new_detection = Detection(
        image_name=image_path,
        detection_score=bbox_score,
        recognized_text=concatenated_text,
        text_score=text_score,
        date=date_time
    )
    session.add(new_detection)
    session.commit()
    session.close()
    print("Detection added successfully!")

# Exemple d'utilisation avec des données fictives
image_path = "path/to/image1.jpg"
bbox_score = 0.95
concatenated_text = "ABC123"
text_score = 0.99

record_detection(image_path, bbox_score, concatenated_text, text_score)
