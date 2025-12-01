# database.py (Ensure this structure is used!)

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

engine = create_engine('sqlite:///health_db.db')
Base = declarative_base()

class HealthRecord(Base):
    """Health Data Entry Module ka table."""
    __tablename__ = 'health_records'
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False) 
    timestamp = Column(DateTime, default=datetime.utcnow) 
    weight = Column(Float)
    systolic_bp = Column(Float)
    diastolic_bp = Column(Float)
    sugar_level = Column(Float)
    sleep_hours = Column(Float)
    exercise_minutes = Column(Float)
    stress_level = Column(Integer)
    
class UserCredentials(Base):
    """Login Credentials ka table."""
    __tablename__ = 'user_credentials'
    id = Column(Integer, primary_key=True)
    # ZAROORI: 'email' field yahan hona chahiye
    email = Column(String, unique=True, nullable=False) 
    password_hash = Column(String, nullable=False) 
    username = Column(String, nullable=False) 
    role = Column(String, default='user')

def initialize_database():
    """Database file aur uske andar ke tables ko create karta hai."""
    try:
        Base.metadata.create_all(engine)
        print("Database 'health_db.db' aur tables successfully initialize ho chuke hain.")
    except Exception as e:
        print(f"Database initialization mein error aagaya: {e}")

Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    initialize_database()