"""
Database Schemas for Phoenix Virtual Assistant

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name (e.g., User -> "user").
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Core user profile and preferences
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    preferences: dict = Field(default_factory=dict, description="Key/value preferences for personalization")
    avatar_url: Optional[str] = Field(None, description="Profile avatar URL")
    is_active: bool = Field(True, description="Whether user is active")

# Conversation session metadata
class Session(BaseModel):
    user_id: Optional[str] = Field(None, description="Associated user id (stringified ObjectId)")
    title: str = Field(default="New Session", description="Session display title")
    sentiment: Optional[Literal["positive","neutral","negative"]] = Field(None, description="Aggregated session sentiment")
    last_message_at: Optional[datetime] = Field(None, description="Timestamp of the last message in this session")

# Individual chat messages
class Message(BaseModel):
    session_id: str = Field(..., description="Related session id (stringified ObjectId)")
    role: Literal["user","assistant","system"] = Field(..., description="Message author role")
    content: str = Field(..., description="Plain text content of the message")
    emotions: Optional[List[str]] = Field(default=None, description="Optional detected emotions for this turn")
    sentiment: Optional[Literal["positive","neutral","negative"]] = Field(default=None, description="Sentiment label for this message")

# Ingested documents for RAG
class Document(BaseModel):
    user_id: Optional[str] = Field(None, description="Owner user id")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Raw extracted text")
    tags: List[str] = Field(default_factory=list, description="Labels for filtering/searching")
    source: Optional[str] = Field(None, description="Where this doc came from (url, upload, etc.)")
