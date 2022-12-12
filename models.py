from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Users(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, onupdate=datetime.now())

    predictions = relationship("Predictions", back_populates="owner")

class Predictions(Base):
    __tablename__ = "predictions_tbl"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    predict = Column(String)
    feedback = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, onupdate=datetime.now())
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    owner = relationship("Users", back_populates="predictions")
