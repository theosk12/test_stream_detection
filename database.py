# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///detections.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, index=True)
    recognized_text = Column(String)
    date = Column(DateTime, default=datetime.utcnow)




# Définition de la table WebcamDetection
class WebcamDetection(Base):
    __tablename__ = "webcam_detections"
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, index=True)
    recognized_text = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

class VideoDetection(Base):
    __tablename__ = "video_detections"
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, index=True)
    recognized_text = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
