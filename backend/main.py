"""
ReviewScope Backend API
FastAPI сервер для анализа отзывов WB/Onliner
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
import zipfile
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import uvicorn
from dotenv import load_dotenv
import requests

# Database imports - handle both module and script execution
try:
    from .database import get_db, init_db, SessionLocal
    from .repository import AnalysisRepository, UserAnalysisRepository, AnalysisViewRepository
    from .models import Analysis
except ImportError:
    from database import get_db, init_db, SessionLocal
    from repository import AnalysisRepository, UserAnalysisRepository, AnalysisViewRepository
    from models import Analysis

# Load .env
load_dotenv()

# ============================================================
# Pydantic Models
# ============================================================

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL товара WB или Onliner")
    use_cache: bool = Field(default=True, description="Использовать кэш если есть")

class ChatRequest(BaseModel):
    session_id: str
    question: str

class AnalysisStatus(BaseModel):
    session_id: str
    status: str  # pending, fetching, scoring, building_rag, summarizing, ready, error
    progress: int  # 0-100
    message: str
    eta_seconds: Optional[int] = None

class ReviewItem(BaseModel):
    id: str
    user: str
    rating: int
    trust: float
    text: str
    created: str
    sentiment: str
    pred_star_soft: Optional[float] = None

class ChatResponse(BaseModel):
    answer: str
    evidence: List[Dict[str, Any]]

# ============================================================
# App Setup
# ============================================================

app = FastAPI(
    title="ReviewScope API",
    description="AI-powered review analysis for WB & Onliner",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (fallback + active tasks)
sessions: Dict[str, Dict[str, Any]] = {}  # For backward compatibility during migration
analysis_tasks: Dict[str, asyncio.Task] = {}

# Flag to use database (can be disabled for testing)
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"

# ============================================================
# Config - пути адаптированы под твою структуру
# ============================================================

# Папка backend/data для хранения результатов
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Repo root
HERE_DIR = Path(__file__).parent
BASE_DIR = HERE_DIR.parent if (HERE_DIR.parent / "reviewscope_all.py").exists() else HERE_DIR
MODELS_DIR = BASE_DIR / "models"

# Путь к reviewscope_all.py - он лежит рядом с backend (в reviewscope/)
REVIEWSCOPE_PATH = BASE_DIR / "reviewscope_all.py"

# Если не нашли, пробуем в корне ReviewScope
if not REVIEWSCOPE_PATH.exists() and (HERE_DIR / "reviewscope_all.py").exists():
    REVIEWSCOPE_PATH = HERE_DIR / "reviewscope_all.py"

print(f"[CONFIG] reviewscope_all.py path: {REVIEWSCOPE_PATH}")
print(f"[CONFIG] exists: {REVIEWSCOPE_PATH.exists()}")

# Модели и API из .env
SENT_MODEL = os.getenv("SENT_MODEL", "pravdorub_sentiment_ru_big_bal")
RATE_MODEL = os.getenv("RATE_MODEL", "pravdorub_rating_ru_max")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
DEVICE = os.getenv("DEVICE", "cpu")

def resolve_model_path(value: str) -> str:
    # Keep HF repo ids as-is; resolve only path-like values.
    if value.startswith(".") or "/" in value or "\\" in value:
        p = Path(value)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return str(p)
    return value

SENT_MODEL = resolve_model_path(SENT_MODEL)
RATE_MODEL = resolve_model_path(RATE_MODEL)

print(f"[CONFIG] SENT_MODEL: {SENT_MODEL}")
print(f"[CONFIG] RATE_MODEL: {RATE_MODEL}")
print(f"[CONFIG] LLM_PROVIDER: {LLM_PROVIDER}")
print(f"[CONFIG] DEVICE: {DEVICE}")

# ============================================================
# Helper Functions
# ============================================================

def _model_files_present(model_dir: str) -> bool:
    p = Path(model_dir)
    if p.is_dir():
        return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return p.exists()

def _extract_gdrive_file_id(url: str) -> Optional[str]:
    # Supports share links like .../file/d/<id>/view and links with ?id=<id>
    if "id=" in url:
        m = re.search(r"[?&]id=([^&]+)", url)
        if m:
            return m.group(1)
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None

def _download_gdrive_file(file_id: str, dest_path: Path) -> None:
    session = requests.Session()
    base_url = "https://docs.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True)

    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

    response.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def _ensure_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False

def ensure_models_available() -> None:
    if _model_files_present(SENT_MODEL) and _model_files_present(RATE_MODEL):
        print("[MODELS] Models already present, skipping download")
        return

    gdrive_url = os.getenv("GDRIVE_URL")
    gdrive_file_id = os.getenv("GDRIVE_FILE_ID")
    if not gdrive_file_id and gdrive_url:
        gdrive_file_id = _extract_gdrive_file_id(gdrive_url)

    if not gdrive_file_id:
        print("[MODELS] Missing models and no GDRIVE_URL/GDRIVE_FILE_ID set")
        return

    zip_name = os.getenv("GDRIVE_ZIP_NAME", "models.zip")
    zip_path = Path(os.getenv("GDRIVE_ZIP_PATH", str(BASE_DIR / zip_name)))

    print(f"[MODELS] Downloading models from Google Drive to {zip_path}")
    _download_gdrive_file(gdrive_file_id, zip_path)

    if not _ensure_zip(zip_path):
        print("[MODELS] Downloaded file is not a zip, retrying with gdown")
        try:
            import gdown  # type: ignore
        except Exception as exc:
            print(f"[MODELS] gdown not available: {exc}")
            return
        # gdown expects just the file id
        gdown.download(id=gdrive_file_id, output=str(zip_path), quiet=False)

    if not _ensure_zip(zip_path):
        print("[MODELS] Download failed: not a zip file")
        return

    print(f"[MODELS] Extracting {zip_path} into {BASE_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BASE_DIR)

    if _model_files_present(SENT_MODEL) and _model_files_present(RATE_MODEL):
        print("[MODELS] Models ready")
    else:
        print("[MODELS] Download finished but model files not found")

def url_to_cache_key(url: str) -> str:
    return hashlib.md5(url.strip().lower().encode()).hexdigest()[:16]

def detect_service(url: str) -> str:
    s = url.strip().lower()
    if "wildberries." in s or "wb.ru" in s:
        return "wb"
    if "onliner.by" in s:
        return "onliner"
    if s.isdigit() and len(s) >= 6:
        return "wb"
    return "onliner"

def sentiment_from_probs(neg: float, neu: float, pos: float) -> str:
    if pos > neg and pos > neu:
        return "pos"
    if neg > pos and neg > neu:
        return "neg"
    return "neu"

def format_review_for_frontend(r: Dict[str, Any]) -> Dict[str, Any]:
    sent = sentiment_from_probs(
        r.get("sent_neg", 0),
        r.get("sent_neu", 0),
        r.get("sent_pos", 1)
    )
    return {
        "id": str(r.get("id", "")),
        "user": r.get("user", "Аноним") or "Аноним",
        "rating": int(r.get("orig_star", 0) or r.get("pred_star_hard", 3)),
        "trust": round(float(r.get("trust", 0.5)), 3),
        "text": r.get("text", ""),
        "created": r.get("created", "")[:10] if r.get("created") else "",
        "sentiment": sent,
        "pred_star_soft": r.get("pred_star_soft")
    }

def load_reviewscope_module():
    """Динамически загружаем reviewscope_all.py"""
    import importlib.util
    import sys

    if not REVIEWSCOPE_PATH.exists():
        raise RuntimeError(f"reviewscope_all.py not found at {REVIEWSCOPE_PATH}")

    # Проверяем, не загружен ли уже
    if "reviewscope_all" in sys.modules:
        return sys.modules["reviewscope_all"]

    spec = importlib.util.spec_from_file_location("reviewscope_all", REVIEWSCOPE_PATH)
    rs = importlib.util.module_from_spec(spec)

    # ВАЖНО: добавляем в sys.modules ДО exec_module
    sys.modules["reviewscope_all"] = rs

    spec.loader.exec_module(rs)
    return rs

@app.on_event("startup")
async def startup_event():
    # Initialize database tables
    if USE_DATABASE:
        try:
            init_db()
            print("[DB] Database initialized successfully")
        except Exception as e:
            print(f"[DB] Warning: Could not initialize database: {e}")

    # Download models in background so healthcheck can pass quickly.
    asyncio.create_task(asyncio.to_thread(ensure_models_available))

# ============================================================
# Background Analysis Task
# ============================================================

def update_analysis_status(session_id: str, status: str, progress: int, message: str, error: str = None):
    """Update status in both memory and database"""
    # Update in-memory
    if session_id in sessions:
        sessions[session_id]["status"] = status
        sessions[session_id]["progress"] = progress
        sessions[session_id]["message"] = message
        if error:
            sessions[session_id]["error"] = error

    # Update in database
    if USE_DATABASE:
        try:
            db = SessionLocal()
            AnalysisRepository.update_status(db, session_id, status, progress, message, error)
            db.close()
        except Exception as e:
            print(f"[DB] Warning: Could not update status: {e}")


async def run_analysis(session_id: str, url: str):
    """Фоновая задача анализа"""
    session = sessions.get(session_id, {})

    try:
        rs = load_reviewscope_module()

        out_dir = DATA_DIR / session_id
        out_dir.mkdir(exist_ok=True)

        # Stage 1: Fetching
        update_analysis_status(session_id, "fetching", 10, "Собираем отзывы...")

        service = detect_service(url)
        session["service"] = service

        loop = asyncio.get_event_loop()

        # Stage 3: Build bundle
        update_analysis_status(session_id, "scoring", 30, "Анализируем тональность и доверие...")

        bundle_path = await loop.run_in_executor(
            None,
            lambda: rs.stage3_build_bundle(
                url=url,
                out_dir=out_dir,
                sent_model_dir=SENT_MODEL,
                rate_model_dir=RATE_MODEL,
                device_str=DEVICE,
                min_len_fetch=15,
                threshold=1000,
                per_rating=100,
                min_len=20,
                min_alpha=10,
                batch=32,
                max_len=256,
                topk=8,
                suspicious_thr=0.30,
                debug=False,
            )
        )

        # Load reviews for streaming animation
        try:
            _, raw_reviews = rs.load_stage3_bundle(bundle_path)
            session["raw_reviews"] = raw_reviews
        except Exception:
            session["raw_reviews"] = []

        update_analysis_status(session_id, "building_rag", 60, "Строим поисковый индекс...")

        # Stage 4: RAG build
        rag_dir = out_dir / "rag"

        await loop.run_in_executor(
            None,
            lambda: rs.rag_build_from_bundle(
                bundle_path=bundle_path,
                rag_dir=rag_dir,
                emb_model=EMB_MODEL,
                emb_batch=48
            )
        )

        update_analysis_status(session_id, "summarizing", 80, "Генерируем AI-сводку...")

        # Stage 4: Summarize
        summary_obj, reviews = rs.load_stage3_bundle(bundle_path)
        llm_cfg = rs.LLMProviderConfig(provider=LLM_PROVIDER, model=LLM_MODEL, temperature=0.2)

        stage4_summary = await loop.run_in_executor(
            None,
            lambda: rs.summarize_product(summary_obj, reviews, llm_cfg, max_evidence=90)
        )

        # Save to file
        summary_path = out_dir / "stage4_summary.json"
        rs.write_json(summary_path, stage4_summary)

        # Store in memory session
        session["bundle_path"] = str(bundle_path)
        session["rag_dir"] = str(rag_dir)
        session["summary_obj"] = summary_obj
        session["stage4_summary"] = stage4_summary
        session["reviews"] = reviews
        session["status"] = "ready"
        session["progress"] = 100
        session["message"] = "Готово!"

        # Store in database
        if USE_DATABASE:
            try:
                db = SessionLocal()
                AnalysisRepository.update_results(
                    db,
                    session_id,
                    summary_data=summary_obj,
                    stage4_summary=stage4_summary,
                    bundle_path=str(bundle_path),
                    rag_dir=str(rag_dir)
                )
                db.close()
            except Exception as e:
                print(f"[DB] Warning: Could not save results: {e}")

        print(f"[OK] Analysis complete for {session_id}")

    except Exception as e:
        import traceback
        error_msg = str(e)
        session["status"] = "error"
        session["message"] = f"Ошибка: {error_msg}"
        session["error"] = error_msg
        session["traceback"] = traceback.format_exc()

        update_analysis_status(session_id, "error", 0, f"Ошибка: {error_msg}", error_msg)

        print(f"[ERROR] {session_id}: {e}")
        print(traceback.format_exc())

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "service": "ReviewScope API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "reviewscope_path": str(REVIEWSCOPE_PATH),
        "reviewscope_exists": REVIEWSCOPE_PATH.exists(),
        "config": {
            "sent_model": SENT_MODEL,
            "rate_model": RATE_MODEL,
            "llm_provider": LLM_PROVIDER,
            "device": DEVICE,
        }
    }

@app.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Запускает анализ товара"""

    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL не может быть пустым")

    cache_key = url_to_cache_key(url)
    service = detect_service(url)

    # Check cache in database first
    if req.use_cache and USE_DATABASE:
        try:
            db = SessionLocal()
            cached = AnalysisRepository.get_by_cache_key(db, cache_key, status="ready")
            if cached:
                # Load into memory if not present
                if cached.id not in sessions:
                    sessions[cached.id] = {
                        "url": cached.url,
                        "cache_key": cached.cache_key,
                        "status": "ready",
                        "progress": 100,
                        "message": "Загружено из кэша",
                        "service": cached.service,
                        "bundle_path": cached.bundle_path,
                        "rag_dir": cached.rag_dir,
                        "summary_obj": cached.summary_data,
                        "stage4_summary": cached.stage4_summary,
                        "reviews": [],  # Will be loaded on demand
                    }
                db.close()
                return AnalysisStatus(
                    session_id=cached.id,
                    status="ready",
                    progress=100,
                    message="Загружено из кэша"
                )
            db.close()
        except Exception as e:
            print(f"[DB] Warning: Cache check failed: {e}")

    # Check in-memory cache
    if req.use_cache:
        for sid, sess in sessions.items():
            if sess.get("cache_key") == cache_key and sess.get("status") == "ready":
                return AnalysisStatus(
                    session_id=sid,
                    status="ready",
                    progress=100,
                    message="Загружено из кэша"
                )

    # New session
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "url": url,
        "cache_key": cache_key,
        "status": "pending",
        "progress": 0,
        "message": "Запуск анализа...",
        "created_at": datetime.now().isoformat(),
        "service": service,
    }

    # Create in database
    if USE_DATABASE:
        try:
            db = SessionLocal()
            AnalysisRepository.create(db, session_id, url, cache_key, service)
            db.close()
        except Exception as e:
            print(f"[DB] Warning: Could not create analysis record: {e}")

    # Start background task
    task = asyncio.create_task(run_analysis(session_id, url))
    analysis_tasks[session_id] = task

    return AnalysisStatus(
        session_id=session_id,
        status="pending",
        progress=0,
        message="Анализ запущен",
        eta_seconds=60
    )

@app.get("/status/{session_id}", response_model=AnalysisStatus)
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")

    sess = sessions[session_id]
    return AnalysisStatus(
        session_id=session_id,
        status=sess["status"],
        progress=sess["progress"],
        message=sess["message"],
        eta_seconds=max(0, int((100 - sess["progress"]) * 0.6)) if sess["status"] not in ("ready", "error") else None
    )

@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")

    sess = sessions[session_id]

    if sess["status"] == "error":
        raise HTTPException(500, sess.get("message", "Unknown error"))

    if sess["status"] != "ready":
        raise HTTPException(400, f"Анализ не завершен. Статус: {sess['status']}")

    summary_obj = sess.get("summary_obj", {})
    stage4 = sess.get("stage4_summary", {})
    reviews = sess.get("reviews", [])

    formatted_reviews = [format_review_for_frontend(r) for r in reviews[:50]]

    product = {
        "url": sess["url"],
        "service": sess.get("service", detect_service(sess["url"])),
        "truthStars": summary_obj.get("stars", {}).get("truth_stars", 0),
        "avgOrig": summary_obj.get("stars", {}).get("avg_orig", 0),
        "suspiciousShare": summary_obj.get("trust", {}).get("suspicious_share", 0),
        "totalReviews": summary_obj.get("counts", {}).get("raw", 0),
        "keptReviews": summary_obj.get("counts", {}).get("kept", 0),
        "sentimentMix": summary_obj.get("sentiment_weighted", {"neg": 0.1, "neu": 0.2, "pos": 0.7}),
    }

    return {
        "session_id": session_id,
        "product": product,
        "summary": stage4,
        "reviews": formatted_reviews,
    }

@app.post("/chat/{session_id}")
async def chat(session_id: str, req: ChatRequest):
    if session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")

    sess = sessions[session_id]

    if sess["status"] != "ready":
        raise HTTPException(400, "Анализ не завершен")

    rag_dir = Path(sess["rag_dir"])

    if not rag_dir.exists():
        raise HTTPException(500, "RAG индекс не найден")

    rs = load_reviewscope_module()

    loop = asyncio.get_event_loop()
    answer, ctx = await loop.run_in_executor(
        None,
        lambda: rs.ask_with_rag(
            rag_dir=rag_dir,
            question=req.question,
            emb_model=EMB_MODEL,
            emb_batch=48,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            temp=0.2,
            top_k=30,
            rerank_k=12,
            max_chars_each=500
        )
    )

    evidence = []
    for item in ctx[:5]:
        evidence.append({
            "id": str(item.get("id", "")),
            "user": "Покупатель",
            "rating": int(item.get("orig_star", 0) or item.get("pred_star_hard", 3)),
            "trust": round(float(item.get("trust", 0.5)), 3),
            "text": item.get("text", "")[:300],
            "created": str(item.get("created", ""))[:10],
            "sentiment": "pos" if (item.get("pred_star_hard", 3) or 3) >= 4 else "neg" if (item.get("pred_star_hard", 3) or 3) <= 2 else "neu"
        })

    return ChatResponse(answer=answer, evidence=evidence)

@app.get("/reviews/{session_id}")
async def get_reviews(session_id: str, skip: int = 0, limit: int = 20, sort_by: str = "trust"):
    if session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")

    sess = sessions[session_id]

    if sess["status"] != "ready":
        raise HTTPException(400, "Анализ не завершен")

    reviews = sess.get("reviews", [])

    if sort_by == "trust":
        reviews = sorted(reviews, key=lambda x: x.get("trust", 0), reverse=True)
    elif sort_by == "rating_high":
        reviews = sorted(reviews, key=lambda x: x.get("orig_star", 0) or 0, reverse=True)
    elif sort_by == "rating_low":
        reviews = sorted(reviews, key=lambda x: x.get("orig_star", 5) or 5)

    paginated = reviews[skip:skip + limit]
    formatted = [format_review_for_frontend(r) for r in paginated]

    return {
        "total": len(sess.get("reviews", [])),
        "skip": skip,
        "limit": limit,
        "reviews": formatted
    }

@app.get("/sessions")
async def list_sessions():
    return {
        sid: {
            "url": s.get("url"),
            "status": s.get("status"),
            "progress": s.get("progress"),
        }
        for sid, s in sessions.items()
    }


@app.get("/reviews-stream/{session_id}")
async def get_reviews_stream(session_id: str, last_seen: int = 0):
    """Get reviews collected so far during analysis (for flying reviews animation)"""
    if session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")

    sess = sessions[session_id]
    raw_reviews = sess.get("raw_reviews", [])

    # Return only new reviews since last_seen index
    new_reviews = raw_reviews[last_seen:last_seen + 10]

    formatted = []
    for r in new_reviews:
        text = r.get("text", "")
        # Truncate for animation display
        if len(text) > 150:
            text = text[:150] + "..."
        formatted.append({
            "id": str(r.get("id", "")),
            "text": text,
            "rating": int(r.get("orig_star", 0) or 0),
            "user": r.get("user", "Покупатель") or "Покупатель",
        })

    return {
        "reviews": formatted,
        "total": len(raw_reviews),
        "next_index": last_seen + len(new_reviews),
        "status": sess.get("status", "pending"),
    }


# ============================================================
# History & Recent Analyses (Database-powered)
# ============================================================

@app.get("/analyses/recent")
async def get_recent_analyses(limit: int = 20, offset: int = 0):
    """Get recently completed analyses (for homepage)"""
    if not USE_DATABASE:
        # Fallback to in-memory
        ready_sessions = [
            {
                "session_id": sid,
                "url": s.get("url"),
                "service": s.get("service"),
                "truth_stars": s.get("summary_obj", {}).get("stars", {}).get("truth_stars"),
                "total_reviews": s.get("summary_obj", {}).get("counts", {}).get("raw"),
                "product_name": s.get("stage4_summary", {}).get("one_liner", "")[:100],
                "created_at": s.get("created_at"),
            }
            for sid, s in sessions.items()
            if s.get("status") == "ready"
        ]
        return {
            "total": len(ready_sessions),
            "analyses": ready_sessions[offset:offset + limit]
        }

    try:
        db = SessionLocal()
        analyses = AnalysisRepository.get_recent(db, limit=limit, offset=offset)
        total = AnalysisRepository.count_ready(db)
        db.close()

        return {
            "total": total,
            "analyses": [
                {
                    "session_id": a.id,
                    "url": a.url,
                    "service": a.service,
                    "truth_stars": a.truth_stars,
                    "avg_orig_stars": a.avg_orig_stars,
                    "total_reviews": a.total_reviews,
                    "kept_reviews": a.kept_reviews,
                    "product_name": a.product_name[:100] if a.product_name else None,
                    "sentiment_mix": a.sentiment_mix,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in analyses
            ]
        }
    except Exception as e:
        print(f"[DB] Error fetching recent analyses: {e}")
        raise HTTPException(500, "Ошибка получения истории")


@app.get("/analysis/{session_id}/meta")
async def get_analysis_meta(session_id: str, request: Request):
    """Get analysis metadata for SEO/sharing (lightweight endpoint)"""
    # Try database first
    if USE_DATABASE:
        try:
            db = SessionLocal()
            analysis = AnalysisRepository.get_by_id(db, session_id)
            if analysis and analysis.status == "ready":
                # Record view
                ip_hash = hashlib.md5(
                    (request.client.host or "unknown").encode()
                ).hexdigest()[:16]
                AnalysisViewRepository.record_view(
                    db, session_id,
                    ip_hash=ip_hash,
                    user_agent=request.headers.get("user-agent"),
                    referer=request.headers.get("referer")
                )
                db.close()

                return {
                    "session_id": analysis.id,
                    "url": analysis.url,
                    "service": analysis.service,
                    "product_name": analysis.product_name,
                    "truth_stars": analysis.truth_stars,
                    "total_reviews": analysis.total_reviews,
                    "one_liner": analysis.stage4_summary.get("one_liner") if analysis.stage4_summary else None,
                    "score": analysis.stage4_summary.get("score") if analysis.stage4_summary else None,
                }
            db.close()
        except Exception as e:
            print(f"[DB] Error fetching meta: {e}")

    # Fallback to in-memory
    if session_id in sessions and sessions[session_id].get("status") == "ready":
        sess = sessions[session_id]
        stage4 = sess.get("stage4_summary", {})
        summary_obj = sess.get("summary_obj", {})
        return {
            "session_id": session_id,
            "url": sess.get("url"),
            "service": sess.get("service"),
            "product_name": stage4.get("one_liner", "")[:100],
            "truth_stars": summary_obj.get("stars", {}).get("truth_stars"),
            "total_reviews": summary_obj.get("counts", {}).get("raw"),
            "one_liner": stage4.get("one_liner"),
            "score": stage4.get("score"),
        }

    raise HTTPException(404, "Анализ не найден")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8888"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
