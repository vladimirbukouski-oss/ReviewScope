#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for fetching Wildberries product questions
Based on the review fetching pattern from reviewscope_all.py
"""

import json
import re
import requests
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


# ============================================================
# Helper functions (from reviewscope_all.py)
# ============================================================

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", flags=re.U)


def eprint(*a: Any) -> None:
    print(*a)


def session_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }


def req_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Tuple[float, float] = (5.0, 25.0),
    tries: int = 4,
    sleep_base: float = 0.5,
    debug: bool = False,
) -> Any:
    import time
    import random
    
    last_err: Optional[Exception] = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=session_headers())
            if debug:
                eprint(f"[GET] {r.status_code} {url} params={params}")
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else (sleep_base * (1.7 ** attempt) + random.random())
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = sleep_base * (1.7 ** attempt) + random.random()
            if debug:
                eprint(f"[ERR] attempt {attempt}/{tries} wait {wait:.2f}s err={repr(e)} url={url}")
            time.sleep(wait)
    raise RuntimeError(f"Failed GET {url} params={params} err={last_err}")


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s.isdigit():
            return int(s)
        return None
    except Exception:
        return None


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00a0", " ").strip())


# ============================================================
# Wildberries helpers (from reviewscope_all.py)
# ============================================================

def wb_parse_nm_id(url_or_id: str) -> int:
    s = url_or_id.strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"/catalog/(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"nmId=(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{6,})", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Не смог распарсить nmId из: {url_or_id}")


def wb_nm_to_vol_part(nm_id: int) -> Tuple[int, int]:
    vol = nm_id // 100000
    part = nm_id // 1000
    return vol, part


def wb_crc16_arc(num: int) -> int:
    t = num.to_bytes(8, byteorder="little", signed=False)
    n = 0
    for b in t:
        n ^= b
        for _ in range(8):
            n = (n >> 1) ^ 0xA001 if (n & 1) else (n >> 1)
    return n


def wb_feedbacks_partition_host(imt_id: int) -> str:
    partition = "2" if (wb_crc16_arc(int(imt_id)) % 100) >= 50 else "1"
    return f"feedbacks{partition}.wb.ru"


def wb_fetch_card_json(nm_id: int, debug: bool = False) -> Dict[str, Any]:
    """Fetch card.json to get imt_id"""
    vol, part = wb_nm_to_vol_part(nm_id)
    
    # Try card.wb.ru API first (simpler)
    try:
        url = "https://card.wb.ru/cards/v4/detail"
        params = {"dest": -1257786, "locale": "ru", "nm": nm_id}
        return req_json(url, params=params, tries=2, sleep_base=0.25, timeout=(4.0, 15.0), debug=debug)
    except Exception as e:
        if debug:
            eprint(f"[wb] card.wb.ru failed: {e}")
    
    # Fallback to basket hosts
    try:
        url = f"https://basket-{vol:02d}.wb.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
        return req_json(url, tries=2, sleep_base=0.3, timeout=(4.0, 15.0), debug=debug)
    except Exception as e:
        if debug:
            eprint(f"[wb] basket host failed: {e}")
        raise RuntimeError(f"Failed to fetch WB card for nmId={nm_id}: {e}")


def wb_extract_imt_id(card_js: Dict[str, Any]) -> int:
    """Extract imt_id from card.json"""
    for k in ("imtId", "imt_id", "imt", "root"):
        v = card_js.get(k)
        if v is not None and str(v).isdigit():
            return int(v)
    if isinstance(card_js.get("data"), dict):
        prods = card_js["data"].get("products")
        if isinstance(prods, list) and prods:
            for k in ("imtId", "imt_id", "imt", "root"):
                v = prods[0].get(k)
                if v is not None and str(v).isdigit():
                    return int(v)
    if isinstance(card_js.get("products"), list) and card_js["products"]:
        p0 = card_js["products"][0]
        if isinstance(p0, dict):
            for k in ("imtId", "imt_id", "imt", "root"):
                v = p0.get(k)
                if v is not None and str(v).isdigit():
                    return int(v)
    raise RuntimeError("Failed to extract imtId from card.json")


def wb_extract_product_info(card_js: Dict[str, Any]) -> Dict[str, Any]:
    """Extract product info from card.json"""
    info: Dict[str, Any] = {
        "name": None,
        "brand": None,
        "description": None,
        "category": None,
    }
    
    # Try direct fields
    for name_key in ("imt_name", "subj_name", "name", "title", "nm_name"):
        v = card_js.get(name_key)
        if v and isinstance(v, str) and v.strip():
            info["name"] = v.strip()
            break
    
    for brand_key in ("brand", "brand_name", "selling", "supplier"):
        v = card_js.get(brand_key)
        if v and isinstance(v, str) and v.strip():
            info["brand"] = v.strip()
            break
    
    for desc_key in ("description", "desc", "full_description"):
        v = card_js.get(desc_key)
        if v and isinstance(v, str) and v.strip():
            info["description"] = v.strip()
            break
    
    for cat_key in ("subj_root_name", "category", "subject"):
        v = card_js.get(cat_key)
        if v and isinstance(v, str) and v.strip():
            info["category"] = v.strip()
            break
    
    # Try nested structure
    products = None
    if isinstance(card_js.get("data"), dict):
        products = card_js["data"].get("products")
    elif isinstance(card_js.get("products"), list):
        products = card_js["products"]
    
    if isinstance(products, list) and products:
        p0 = products[0]
        if isinstance(p0, dict):
            if not info["name"]:
                for name_key in ("name", "imt_name", "subj_name", "title"):
                    v = p0.get(name_key)
                    if v and isinstance(v, str) and v.strip():
                        info["name"] = v.strip()
                        break
            if not info["brand"]:
                for brand_key in ("brand", "brand_name", "selling"):
                    v = p0.get(brand_key)
                    if v and isinstance(v, str) and v.strip():
                        info["brand"] = v.strip()
                        break
    
    return info


# ============================================================
# Questions fetching (NEW)
# ============================================================

QUESTIONS_PER_PAGE = 30
QUESTIONS_URL = "https://questions.wildberries.ru/api/v1/questions"


def wb_simplify_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify question object to match review format"""
    # Get user name
    user = ""
    if isinstance(q.get("user"), dict):
        user = (q["user"].get("name") or "").strip()
    elif isinstance(q.get("userName"), str):
        user = q.get("userName", "").strip()
    
    # Get question text
    text = (q.get("text") or q.get("question") or "").strip()
    
    # Get answer if exists
    answer = ""
    answer_user = ""
    answer_date = None
    
    if isinstance(q.get("answer"), dict):
        answer_obj = q["answer"]
        answer = (answer_obj.get("text") or "").strip()
        if isinstance(answer_obj.get("user"), dict):
            answer_user = answer_obj["user"].get("name", "").strip()
        answer_date = answer_obj.get("createdDate") or answer_obj.get("created")
    
    return {
        "id": q.get("id"),
        "created": q.get("createdDate") or q.get("created"),
        "user": user,
        "text": text,
        "answer": answer,
        "answer_user": answer_user,
        "answer_date": answer_date,
        "type": "question",  # Mark as question type
    }


def fetch_wb_questions(
    url: str,
    min_len: int = 10,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch WB questions and product info. Returns (questions, product_info)."""
    
    nm_id = wb_parse_nm_id(url)
    eprint(f"[wb] nmId={nm_id}")
    
    s = requests.Session()
    card = wb_fetch_card_json(nm_id, debug=debug)
    imt_id = wb_extract_imt_id(card)
    product_info = wb_extract_product_info(card)
    eprint(f"[wb] imtId={imt_id}")
    if product_info.get("name"):
        eprint(f"[wb] product: {product_info.get('brand', '')} - {product_info['name']}")
    
    # First, get total count of questions
    try:
        count_params = {"imtId": imt_id, "onlyCount": True}
        count_response = req_json(
            QUESTIONS_URL,
            params=count_params,
            tries=2,
            sleep_base=0.3,
            timeout=(4.0, 15.0),
            debug=debug,
        )
        total_questions = count_response.get("count", 0)
        eprint(f"[wb] total questions: {total_questions}")
    except Exception as e:
        eprint(f"[wb] failed to get question count: {e}")
        total_questions = 0
    
    if total_questions == 0:
        eprint("[wb] no questions found")
        return [], product_info
    
    # Calculate total pages
    total_pages = (total_questions + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE
    eprint(f"[wb] fetching {total_questions} questions from {total_pages} pages")
    
    # Fetch all questions
    rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    
    for page in range(1, total_pages + 1):
        skip = (page - 1) * QUESTIONS_PER_PAGE
        
        try:
            params = {
                "imtId": imt_id,
                "skip": skip,
                "take": QUESTIONS_PER_PAGE,
            }
            payload = req_json(
                QUESTIONS_URL,
                params=params,
                tries=2,
                sleep_base=0.4,
                timeout=(4.0, 25.0),
                debug=debug,
            )
            
            questions = payload.get("questions", [])
            if not questions:
                eprint(f"[wb] page {page}: no questions, stopping")
                break
            
            added_this = 0
            for q in questions:
                qid = q.get("id")
                if qid is not None:
                    if qid in seen_ids:
                        continue
                    seen_ids.add(qid)
                
                row = wb_simplify_question(q)
                
                # Filter by length
                if min_len and len(row["text"]) < min_len:
                    continue
                
                rows.append(row)
                added_this += 1
            
            eprint(f"[wb] page {page}/{total_pages}: fetched {len(questions)} questions, kept {added_this}")
            
            # If we got fewer questions than requested, we're done
            if len(questions) < QUESTIONS_PER_PAGE:
                eprint(f"[wb] page {page}: got fewer questions than requested, stopping")
                break
            
        except Exception as e:
            eprint(f"[wb] error fetching page {page}: {e}")
            continue
    
    eprint(f"[wb] collected {len(rows)} questions")
    return rows, product_info


# ============================================================
# Main
# ============================================================

def main():
    import sys
    
    # Test URL
    test_url = "https://www.wildberries.by/catalog/50252349/detail.aspx?targetUrl=EX"
    
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    eprint("=" * 60)
    eprint("Testing Wildberries Questions Fetch")
    eprint("=" * 60)
    eprint(f"URL: {test_url}")
    eprint()
    
    try:
        questions, product_info = fetch_wb_questions(test_url, min_len=10, debug=True)
        
        eprint()
        eprint("=" * 60)
        eprint(f"Product Info:")
        eprint("=" * 60)
        eprint(f"Name: {product_info.get('name')}")
        eprint(f"Brand: {product_info.get('brand')}")
        eprint(f"Category: {product_info.get('category')}")
        eprint()
        
        eprint("=" * 60)
        eprint(f"Questions ({len(questions)} total):")
        eprint("=" * 60)
        
        for i, q in enumerate(questions[:10], 1):  # Show first 10
            eprint()
            eprint(f"--- Question {i} ---")
            eprint(f"ID: {q['id']}")
            eprint(f"User: {q['user'] or 'Anonymous'}")
            eprint(f"Date: {q['created']}")
            eprint(f"Question: {q['text'][:200]}...")
            if q['answer']:
                eprint(f"Answer: {q['answer'][:200]}...")
                eprint(f"Answer by: {q['answer_user'] or 'Seller'}")
            else:
                eprint("Answer: No answer yet")
        
        if len(questions) > 10:
            eprint()
            eprint(f"... and {len(questions) - 10} more questions")
        
        # Save to JSON
        output_file = "test_questions_output.json"
        output_data = {
            "product_info": product_info,
            "questions": questions,
            "total": len(questions),
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        eprint()
        eprint("=" * 60)
        eprint(f"Results saved to: {output_file}")
        eprint("=" * 60)
        
    except Exception as e:
        eprint(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
