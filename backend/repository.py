"""
ReviewScope Repository Layer
Database operations for Analysis, User, and TrackedProduct
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

try:
    from .models import Analysis, User, UserAnalysis, TrackedProduct, AnalysisView
except ImportError:
    from models import Analysis, User, UserAnalysis, TrackedProduct, AnalysisView


class AnalysisRepository:
    """CRUD operations for Analysis"""

    @staticmethod
    def create(db: Session, analysis_id: str, url: str, cache_key: str, service: str) -> Analysis:
        """Create a new analysis record"""
        analysis = Analysis(
            id=analysis_id,
            url=url,
            cache_key=cache_key,
            service=service,
            status="pending",
            progress=0,
            message="Запуск анализа...",
            created_at=datetime.utcnow()
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis

    @staticmethod
    def get_by_id(db: Session, analysis_id: str) -> Optional[Analysis]:
        """Get analysis by ID"""
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()

    @staticmethod
    def get_by_cache_key(db: Session, cache_key: str, status: str = "ready") -> Optional[Analysis]:
        """Find cached analysis by URL hash"""
        return db.query(Analysis).filter(
            and_(Analysis.cache_key == cache_key, Analysis.status == status)
        ).order_by(desc(Analysis.created_at)).first()

    @staticmethod
    def update_status(
        db: Session,
        analysis_id: str,
        status: str,
        progress: int,
        message: str,
        error: Optional[str] = None
    ) -> Optional[Analysis]:
        """Update analysis status and progress"""
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.status = status
            analysis.progress = progress
            analysis.message = message
            if error:
                analysis.error = error
            analysis.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(analysis)
        return analysis

    @staticmethod
    def update_results(
        db: Session,
        analysis_id: str,
        summary_data: Dict[str, Any],
        stage4_summary: Dict[str, Any],
        bundle_path: str,
        rag_dir: str
    ) -> Optional[Analysis]:
        """Update analysis with final results"""
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.summary_data = summary_data
            analysis.stage4_summary = stage4_summary
            analysis.bundle_path = bundle_path
            analysis.rag_dir = rag_dir

            # Extract key metrics
            if summary_data:
                stars = summary_data.get("stars", {})
                analysis.truth_stars = stars.get("truth_stars")
                analysis.avg_orig_stars = stars.get("avg_orig")
                analysis.suspicious_share = summary_data.get("trust", {}).get("suspicious_share")
                counts = summary_data.get("counts", {})
                analysis.total_reviews = counts.get("raw")
                analysis.kept_reviews = counts.get("kept")
                analysis.sentiment_mix = summary_data.get("sentiment_weighted")

            # Extract product name from summary
            if stage4_summary:
                analysis.product_name = stage4_summary.get("one_liner", "")[:500]

            analysis.status = "ready"
            analysis.progress = 100
            analysis.message = "Готово!"
            analysis.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(analysis)
        return analysis

    @staticmethod
    def get_recent(db: Session, limit: int = 20, offset: int = 0) -> List[Analysis]:
        """Get recent completed analyses"""
        return db.query(Analysis).filter(
            Analysis.status == "ready"
        ).order_by(desc(Analysis.created_at)).offset(offset).limit(limit).all()

    @staticmethod
    def count_ready(db: Session) -> int:
        """Count completed analyses"""
        return db.query(Analysis).filter(Analysis.status == "ready").count()

    @staticmethod
    def delete(db: Session, analysis_id: str) -> bool:
        """Delete analysis by ID"""
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            db.delete(analysis)
            db.commit()
            return True
        return False


class UserRepository:
    """CRUD operations for User"""

    @staticmethod
    def create(db: Session, email: str, password_hash: str, name: Optional[str] = None) -> User:
        """Create a new user"""
        user = User(
            email=email.lower().strip(),
            password_hash=password_hash,
            name=name,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email.lower().strip()).first()

    @staticmethod
    def update_last_login(db: Session, user_id: int) -> Optional[User]:
        """Update user's last login timestamp"""
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.last_login = datetime.utcnow()
            db.commit()
            db.refresh(user)
        return user


class UserAnalysisRepository:
    """Operations for user-analysis relationships"""

    @staticmethod
    def add_to_history(db: Session, user_id: int, analysis_id: str) -> UserAnalysis:
        """Add analysis to user's history"""
        # Check if already exists
        existing = db.query(UserAnalysis).filter(
            and_(UserAnalysis.user_id == user_id, UserAnalysis.analysis_id == analysis_id)
        ).first()

        if existing:
            existing.viewed_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing

        ua = UserAnalysis(
            user_id=user_id,
            analysis_id=analysis_id,
            viewed_at=datetime.utcnow()
        )
        db.add(ua)
        db.commit()
        db.refresh(ua)
        return ua

    @staticmethod
    def toggle_favorite(db: Session, user_id: int, analysis_id: str) -> bool:
        """Toggle favorite status for an analysis"""
        ua = db.query(UserAnalysis).filter(
            and_(UserAnalysis.user_id == user_id, UserAnalysis.analysis_id == analysis_id)
        ).first()

        if ua:
            ua.is_favorite = not ua.is_favorite
            db.commit()
            return ua.is_favorite
        return False

    @staticmethod
    def get_user_history(
        db: Session,
        user_id: int,
        limit: int = 50,
        favorites_only: bool = False
    ) -> List[UserAnalysis]:
        """Get user's analysis history"""
        query = db.query(UserAnalysis).filter(UserAnalysis.user_id == user_id)

        if favorites_only:
            query = query.filter(UserAnalysis.is_favorite == True)

        return query.order_by(desc(UserAnalysis.viewed_at)).limit(limit).all()


class TrackedProductRepository:
    """CRUD operations for TrackedProduct"""

    @staticmethod
    def create(
        db: Session,
        user_id: int,
        url: str,
        service: str,
        product_name: Optional[str] = None
    ) -> TrackedProduct:
        """Create a new tracked product"""
        tp = TrackedProduct(
            user_id=user_id,
            url=url,
            service=service,
            product_name=product_name,
            created_at=datetime.utcnow()
        )
        db.add(tp)
        db.commit()
        db.refresh(tp)
        return tp

    @staticmethod
    def get_user_tracked(db: Session, user_id: int, active_only: bool = True) -> List[TrackedProduct]:
        """Get user's tracked products"""
        query = db.query(TrackedProduct).filter(TrackedProduct.user_id == user_id)

        if active_only:
            query = query.filter(TrackedProduct.is_active == True)

        return query.order_by(desc(TrackedProduct.created_at)).all()

    @staticmethod
    def update_state(
        db: Session,
        tracked_id: int,
        price: Optional[float] = None,
        rating: Optional[float] = None,
        review_count: Optional[int] = None,
        analysis_id: Optional[str] = None
    ) -> Optional[TrackedProduct]:
        """Update tracked product state after check"""
        tp = db.query(TrackedProduct).filter(TrackedProduct.id == tracked_id).first()
        if tp:
            if price is not None:
                tp.last_price = price
            if rating is not None:
                tp.last_rating = rating
            if review_count is not None:
                tp.last_review_count = review_count
            if analysis_id:
                tp.latest_analysis_id = analysis_id
            tp.last_check_at = datetime.utcnow()
            db.commit()
            db.refresh(tp)
        return tp

    @staticmethod
    def delete(db: Session, tracked_id: int, user_id: int) -> bool:
        """Delete tracked product (only if owned by user)"""
        tp = db.query(TrackedProduct).filter(
            and_(TrackedProduct.id == tracked_id, TrackedProduct.user_id == user_id)
        ).first()
        if tp:
            db.delete(tp)
            db.commit()
            return True
        return False


class AnalysisViewRepository:
    """Operations for view tracking"""

    @staticmethod
    def record_view(
        db: Session,
        analysis_id: str,
        ip_hash: Optional[str] = None,
        user_agent: Optional[str] = None,
        referer: Optional[str] = None
    ) -> AnalysisView:
        """Record a page view"""
        view = AnalysisView(
            analysis_id=analysis_id,
            ip_hash=ip_hash,
            user_agent=user_agent[:255] if user_agent else None,
            referer=referer[:512] if referer else None,
            viewed_at=datetime.utcnow()
        )
        db.add(view)
        db.commit()
        return view

    @staticmethod
    def count_views(db: Session, analysis_id: str) -> int:
        """Count total views for an analysis"""
        return db.query(AnalysisView).filter(AnalysisView.analysis_id == analysis_id).count()
