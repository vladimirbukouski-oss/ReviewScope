"""
ReviewScope Database Models
SQLAlchemy models for persistent storage
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship
from .database import Base


class Analysis(Base):
    """Stores product analysis results"""
    __tablename__ = "analyses"

    id = Column(String(16), primary_key=True)  # Short UUID
    url = Column(String(512), nullable=False, index=True)
    cache_key = Column(String(32), nullable=False, index=True)  # MD5 hash of URL
    service = Column(String(20), nullable=False)  # 'wb' or 'onliner'

    # Status tracking
    status = Column(String(20), default="pending")  # pending, fetching, scoring, building_rag, summarizing, ready, error
    progress = Column(Integer, default=0)
    message = Column(String(255), default="")
    error = Column(Text, nullable=True)

    # Product info (extracted from analysis)
    product_name = Column(String(512), nullable=True)
    product_image = Column(String(512), nullable=True)
    truth_stars = Column(Float, nullable=True)
    avg_orig_stars = Column(Float, nullable=True)
    suspicious_share = Column(Float, nullable=True)
    total_reviews = Column(Integer, nullable=True)
    kept_reviews = Column(Integer, nullable=True)

    # Full analysis data (JSON)
    summary_data = Column(JSON, nullable=True)  # stage3 summary_obj
    stage4_summary = Column(JSON, nullable=True)  # AI-generated summary
    sentiment_mix = Column(JSON, nullable=True)  # {neg, neu, pos}

    # Storage paths
    bundle_path = Column(String(512), nullable=True)
    rag_dir = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user_analyses = relationship("UserAnalysis", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_analyses_cache_status', 'cache_key', 'status'),
    )


class User(Base):
    """User accounts for sync and notifications"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Profile
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Settings
    notify_email = Column(Boolean, default=True)
    notify_telegram = Column(Boolean, default=False)
    telegram_chat_id = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    analyses = relationship("UserAnalysis", back_populates="user", cascade="all, delete-orphan")
    tracked_products = relationship("TrackedProduct", back_populates="user", cascade="all, delete-orphan")


class UserAnalysis(Base):
    """Links users to their analyses (history, favorites)"""
    __tablename__ = "user_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    analysis_id = Column(String(16), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)

    is_favorite = Column(Boolean, default=False)
    viewed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="analyses")
    analysis = relationship("Analysis", back_populates="user_analyses")

    __table_args__ = (
        Index('ix_user_analyses_user_favorite', 'user_id', 'is_favorite'),
    )


class TrackedProduct(Base):
    """Products tracked for price/review monitoring"""
    __tablename__ = "tracked_products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    url = Column(String(512), nullable=False)
    service = Column(String(20), nullable=False)
    product_name = Column(String(512), nullable=True)

    # Tracking settings
    is_active = Column(Boolean, default=True)
    check_interval_hours = Column(Integer, default=24)

    # Alert thresholds
    notify_price_drop_percent = Column(Float, default=10.0)  # Alert if price drops > X%
    notify_rating_change = Column(Float, default=0.3)  # Alert if rating changes > X
    notify_new_reviews = Column(Integer, default=10)  # Alert if N+ new reviews

    # Last known state
    last_price = Column(Float, nullable=True)
    last_rating = Column(Float, nullable=True)
    last_review_count = Column(Integer, nullable=True)
    last_check_at = Column(DateTime, nullable=True)

    # Latest analysis reference
    latest_analysis_id = Column(String(16), ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="tracked_products")
    latest_analysis = relationship("Analysis", foreign_keys=[latest_analysis_id])


class AnalysisView(Base):
    """Anonymous view tracking for analytics"""
    __tablename__ = "analysis_views"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(16), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)

    # Anonymous tracking
    ip_hash = Column(String(64), nullable=True)  # Hashed IP for uniqueness
    user_agent = Column(String(255), nullable=True)
    referer = Column(String(512), nullable=True)

    viewed_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index('ix_analysis_views_date', 'analysis_id', 'viewed_at'),
    )
