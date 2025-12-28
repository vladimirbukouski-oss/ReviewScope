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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

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

# In-memory storage
sessions: Dict[str, Dict[str, Any]] = {}
analysis_tasks: Dict[str, asyncio.Task] = {}

# ============================================================
# Config - пути адаптированы под твою структуру
# ============================================================

# Папка backend/data для хранения результатов
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Путь к reviewscope_all.py - он лежит рядом с backend (в reviewscope/)
REVIEWSCOPE_PATH = Path(__file__).parent.parent / "reviewscope_all.py"

# Если не нашли, пробуем в корне ReviewScope
if not REVIEWSCOPE_PATH.exists():
    REVIEWSCOPE_PATH = Path(__file__).parent.parent.parent / "reviewscope_all.py"

print(f"[CONFIG] reviewscope_all.py path: {REVIEWSCOPE_PATH}")
print(f"[CONFIG] exists: {REVIEWSCOPE_PATH.exists()}")

# Модели и API из .env
SENT_MODEL = os.getenv("SENT_MODEL", "pravdorub_sentiment_ru_big_bal")
RATE_MODEL = os.getenv("RATE_MODEL", "pravdorub_rating_ru_max")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
DEVICE = os.getenv("DEVICE", "cpu")

print(f"[CONFIG] SENT_MODEL: {SENT_MODEL}")
print(f"[CONFIG] RATE_MODEL: {RATE_MODEL}")
print(f"[CONFIG] LLM_PROVIDER: {LLM_PROVIDER}")
print(f"[CONFIG] DEVICE: {DEVICE}")

# ============================================================
# Helper Functions
# ============================================================

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

# ============================================================
# Background Analysis Task
# ============================================================

async def run_analysis(session_id: str, url: str):
    """Фоновая задача анализа"""
    session = sessions[session_id]

    try:
        rs = load_reviewscope_module()

        out_dir = DATA_DIR / session_id
        out_dir.mkdir(exist_ok=True)

        # Stage 1: Fetching
        session["status"] = "fetching"
        session["progress"] = 10
        session["message"] = "Собираем отзывы..."

        service = detect_service(url)
        session["service"] = service

        loop = asyncio.get_event_loop()

        # Stage 3: Build bundle
        session["status"] = "scoring"
        session["progress"] = 30
        session["message"] = "Анализируем тональность и доверие..."

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

        session["progress"] = 60
        session["message"] = "Строим поисковый индекс..."

        # Stage 4: RAG build
        session["status"] = "building_rag"
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

        session["progress"] = 80
        session["message"] = "Генерируем AI-сводку..."

        # Stage 4: Summarize
        session["status"] = "summarizing"

        summary_obj, reviews = rs.load_stage3_bundle(bundle_path)
        llm_cfg = rs.LLMProviderConfig(provider=LLM_PROVIDER, model=LLM_MODEL, temperature=0.2)

        stage4_summary = await loop.run_in_executor(
            None,
            lambda: rs.summarize_product(summary_obj, reviews, llm_cfg, max_evidence=90)
        )

        # Save
        summary_path = out_dir / "stage4_summary.json"
        rs.write_json(summary_path, stage4_summary)

        # Store in session
        session["bundle_path"] = str(bundle_path)
        session["rag_dir"] = str(rag_dir)
        session["summary_obj"] = summary_obj
        session["stage4_summary"] = stage4_summary
        session["reviews"] = reviews
        session["status"] = "ready"
        session["progress"] = 100
        session["message"] = "Готово!"

        print(f"[OK] Analysis complete for {session_id}")

    except Exception as e:
        import traceback
        session["status"] = "error"
        session["message"] = f"Ошибка: {str(e)}"
        session["error"] = str(e)
        session["traceback"] = traceback.format_exc()
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

    # Check cache
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
    }

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

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)