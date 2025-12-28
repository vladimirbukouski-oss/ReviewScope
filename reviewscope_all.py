#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReviewScope ALL-IN-ONE Pipeline (Stage0..Stage4) — ONE FILE.

What you get in one script:
- Fetch reviews (WB / Onliner) -> reviews_raw.jsonl
- Stage3 scoring (sentiment + rating models) + trust -> stage3_bundle.json + reviews_scored.csv + reviews_ranked.jsonl
- Stage4 RAG: build vector index (HNSW) -> rag_dir (hnsw.index + meta.jsonl)
- Stage4 LLM: ask questions with evidence ids, and generate a structured product summary JSON

Works on Windows (PowerShell) and Linux. Paths can be Windows-style.

Dependencies:
pip install -U requests numpy pandas tqdm hnswlib torch transformers

Env keys:
PowerShell:
  $env:OPENAI_API_KEY="..."
  $env:GROQ_API_KEY="..."   # optional (if you use --llm_provider groq)

Recommended default for Wildberries:
  --fb_from 1 --fb_to 2   (these are the working hosts in your environment)

Commands (single-line examples):
  python reviewscope_all.py run --url "WB_URL_OR_nmId" --out_dir stage3_out --sent_model pravdorub_sentiment_ru_big_bal/final --rate_model pravdorub_rating_ru_max/final --device cpu --fb_from 1 --fb_to 2
  python reviewscope_all.py ask --rag_dir stage3_out/rag --question "Какие основные минусы товара?"
  python reviewscope_all.py summarize --bundle stage3_out/stage3_bundle.json --out stage3_out/stage4_summary.json

If you want Groq ultra-cheap generation:
  python reviewscope_all.py ask --rag_dir stage3_out/rag --question "Есть ли проблемы с размером?" --llm_provider groq --llm_model llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import socket
import sys
import time
import threading
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import hnswlib  # type: ignore
except Exception:
    hnswlib = None


# ============================================================
# Model/tokenizer cache (big speedup on CPU hosting)
# ============================================================

_TOKENIZER_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()


def _env_true(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=16)
def _cached_tokenizer(path: str):
    # Keep compatibility with older transformers; prefer fast tokenizer when available.
    kw = {"use_fast": True}
    sig = inspect.signature(AutoTokenizer.from_pretrained)
    if "fix_mistral_regex" in sig.parameters:
        kw["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(path, **kw)


@lru_cache(maxsize=16)
def _cached_model(path: str, device_key: str, cpu_int8: bool):
    m = AutoModelForSequenceClassification.from_pretrained(path)
    if device_key == "cpu" and cpu_int8:
        # Dynamic INT8 quantization works well for transformer encoders on CPU.
        try:
            m = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
    device = torch.device(device_key)
    return m.to(device).eval()


# ============================================================
# Shared helpers
# ============================================================

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", flags=re.U)


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr)


def session_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(UA_LIST),
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }


def req_json(
    s: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Tuple[float, float] = (5.0, 25.0),
    tries: int = 4,
    sleep_base: float = 0.5,
    debug: bool = False,
) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(1, tries + 1):
        try:
            r = s.get(url, params=params, timeout=timeout, headers=session_headers())
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


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


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


def combined_len(row: Dict[str, Any]) -> int:
    return len((row.get("text") or "")) + len((row.get("pros") or "")) + len((row.get("cons") or ""))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_service(url_or_id: str) -> str:
    s = url_or_id.strip().lower()
    if "wildberries." in s or "wb.ru" in s:
        return "wb"
    if "onliner.by" in s or "catalog.api.onliner.by" in s:
        return "onliner"
    if s.isdigit() and len(s) >= 6:
        return "wb"
    return "onliner"


# ============================================================
# Fetch: Onliner
# ============================================================

ON_REVIEWS_URL = "https://catalog.api.onliner.by/products/{key}/reviews"
ON_STOP_TAILS = {"reviews", "prices", "specs", "description", "offers", "forum"}


def on_parse_product_key(url_or_key: str) -> str:
    s = url_or_key.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]+", s):
        return s
    m = re.search(r"catalog\.api\.onliner\.by/products/([^/?#]+)", s)
    if m:
        return m.group(1)
    p = urlparse(s)
    parts = [x for x in p.path.split("/") if x]
    if not parts:
        raise ValueError(f"Не смог вытащить product key из: {url_or_key}")
    tail = parts[-1].lower()
    if tail in ON_STOP_TAILS and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def on_fetch_reviews_page(s: requests.Session, key: str, page: int, debug: bool = False) -> Dict[str, Any]:
    # IMPORTANT: Onliner /reviews supports limit max 10, иначе 422
    return req_json(
        s,
        ON_REVIEWS_URL.format(key=key),
        params={"page": page, "limit": 10},
        tries=4,
        sleep_base=0.4,
        timeout=(4.0, 25.0),
        debug=debug,
    )


def on_parse_reviews(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        return payload["reviews"]
    return []


def on_simplify_review(r: Dict[str, Any]) -> Dict[str, Any]:
    author = r.get("author") if isinstance(r.get("author"), dict) else {}
    summary = (r.get("summary") or "").strip()
    text = (r.get("text") or "").strip()
    merged = (summary + "\n" + text).strip() if summary and text else (summary or text)
    return {
        "id": r.get("id"),
        "rating": safe_int(r.get("rating")),
        "created": r.get("created_at") or r.get("created"),
        "user": (author.get("name") or "").strip() if isinstance(author, dict) else "",
        "text": merged,
        "pros": (r.get("pros") or "").strip(),
        "cons": (r.get("cons") or "").strip(),
    }


def seed_buckets_from_rows(rows: List[Dict[str, Any]], per_rating: int) -> Dict[int, List[Dict[str, Any]]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for row in rows:
        rt = safe_int(row.get("rating"))
        if rt in buckets and len(buckets[rt]) < per_rating:
            buckets[rt].append(row)
    return buckets


def all_buckets_full(buckets: Dict[int, List[Dict[str, Any]]], per_rating: int) -> bool:
    return all(len(buckets[r]) >= per_rating for r in (1, 2, 3, 4, 5))


def stratified_result(buckets: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rating in (1, 2, 3, 4, 5):
        out.extend(buckets.get(rating, []))
    return out


def fetch_onliner_reviews(
    url: str,
    min_len: int,
    threshold_total: int,
    per_rating: int,
    sleep_s: float,
    debug: bool,
) -> List[Dict[str, Any]]:
    key = on_parse_product_key(url)
    eprint(f"[onliner] key={key}")

    s = requests.Session()
    rows: List[Dict[str, Any]] = []
    buckets: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    sampling_mode = False
    seen_ids: set = set()
    seen_fb: set = set()

    page = 1
    pbar = tqdm(total=None, unit="rev", desc="reviews", disable=False)
    while True:
        payload = on_fetch_reviews_page(s, key, page=page, debug=debug)
        revs = on_parse_reviews(payload)
        if not revs:
            break

        added_this = 0
        for r in revs:
            row = on_simplify_review(r)
            rid = row.get("id")
            if rid is not None:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            else:
                fbk = (str(row.get("created") or ""), str(row.get("rating") or ""), (row.get("text") or "")[:120])
                if fbk in seen_fb:
                    continue
                seen_fb.add(fbk)

            if row["rating"] is None:
                continue
            if min_len and combined_len(row) < min_len:
                continue

            if not sampling_mode:
                rows.append(row)
                added_this += 1
                if len(rows) > threshold_total:
                    sampling_mode = True
                    buckets = seed_buckets_from_rows(rows, per_rating)
                    rows = []
            else:
                rt = int(row["rating"])
                if rt in buckets and len(buckets[rt]) < per_rating:
                    buckets[rt].append(row)
                    added_this += 1

        pbar.update(added_this)
        if sampling_mode and all_buckets_full(buckets, per_rating):
            break
        page += 1
        time.sleep(max(0.0, sleep_s + random.random() * 0.2))
    pbar.close()

    if sampling_mode:
        out = stratified_result(buckets)
        eprint(f"[onliner] sampled={len(out)} (<= {per_rating} per rating)")
        return out
    eprint(f"[onliner] collected={len(rows)}")
    return rows


# ============================================================
# Fetch: Wildberries
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


WB_UPSTREAMS_URL = "https://cdn.wbbasket.ru/api/v3/upstreams"


def wb_nm_to_vol_part(nm_id: int) -> Tuple[int, int]:
    vol = nm_id // 100000
    part = nm_id // 1000
    return vol, part


def wb_get_upstreams(s: requests.Session, debug: bool = False) -> Dict[str, Any]:
    return req_json(s, WB_UPSTREAMS_URL, tries=3, sleep_base=0.3, timeout=(4.0, 12.0), debug=debug)


def wb_pick_host_by_vol(route_hosts: List[Dict[str, Any]], vol: int) -> Optional[str]:
    for h in route_hosts:
        a = h.get("vol_range_from")
        b = h.get("vol_range_to")
        if a is None or b is None:
            continue
        if int(a) <= vol <= int(b):
            return h.get("host")
    return None


def wb_resolve_card_host(s: requests.Session, vol: int, debug: bool = False) -> str:
    """Resolve which *.wbbasket.ru host stores card.json for a given volume.

    WB регулярно меняет схему upstreams. Вместо жёсткой привязки к одному ключу,
    пытаемся найти подходящий route_map в нескольких местах.
    """
    ups = wb_get_upstreams(s, debug=debug)

    # Collect candidate host maps from both "recommend" and "origin" sections.
    candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
    for sect in ("recommend", "origin"):
        sec = ups.get(sect)
        if not isinstance(sec, dict):
            continue
        for key in ("mediabasket_route_map", "basket_route_map"):
            rm = sec.get(key)
            if isinstance(rm, list) and rm and isinstance(rm[0], dict):
                hosts = rm[0].get("hosts")
                if isinstance(hosts, list) and hosts:
                    candidates.append((f"{sect}.{key}", hosts))

    # Try to pick a host from any of the candidates.
    for name, hosts in candidates:
        host = wb_pick_host_by_vol(hosts, vol)
        if host:
            return host
        if debug:
            eprint(f"[wb] no host for vol={vol} in {name}")

    # As a very defensive fallback: if vol is above the max known range, return the host
    # with the largest vol_range_to (better than crashing with a confusing error).
    max_to: Optional[int] = None
    max_host: Optional[str] = None
    for _, hosts in candidates:
        for h in hosts:
            try:
                b = int(h.get("vol_range_to"))
            except Exception:
                continue
            if max_to is None or b > max_to:
                max_to = b
                max_host = h.get("host")
    if max_to is not None and max_host and vol > max_to:
        if debug:
            eprint(f"[wb] vol={vol} выше max_range_to={max_to}; fallback host={max_host}")
        return max_host

    raise RuntimeError(f"Failed to find WB host for vol={vol} in upstreams (no matching route_map).")


def wb_try_fetch_card_json_from_hosts(
    s: requests.Session,
    nm_id: int,
    vol: int,
    part: int,
    hosts: List[str],
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Try to GET card.json from a list of candidate basket hosts (best-effort)."""
    url_path = f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
    for h in hosts:
        try:
            return req_json(
                s,
                f"https://{h}{url_path}",
                tries=1,
                sleep_base=0.2,
                timeout=(4.0, 12.0),
                debug=debug,
            )
        except Exception:
            continue
    return None


def wb_fetch_card_json_via_api(s: requests.Session, nm_id: int, debug: bool = False) -> Dict[str, Any]:
    # Endpoints here are not part of the official seller API, and they change over time.
    # We therefore try a small set of known variants.
    attempts: List[Tuple[str, List[Dict[str, Any]]]] = [
        (
            "https://card.wb.ru/cards/v4/detail",
            [
                {"dest": -1257786, "locale": "ru", "nm": nm_id},
                {"dest": "-1216601,-115136,-421732,123585595", "locale": "ru", "nm": nm_id},
            ],
        ),
        (
            "https://card.wb.ru/cards/v2/detail",
            [
                {"ab_testing": "false", "appType": 1, "curr": "rub", "dest": -1257786, "spp": 30, "nm": nm_id},
                {"appType": 1, "curr": "byn", "dest": -59202, "spp": 30, "nm": nm_id},
            ],
        ),
        (
            "https://card.wb.ru/cards/detail",
            [
                {"appType": 1, "curr": "rub", "dest": -1257786, "spp": 30, "nm": nm_id},
                {"nm": nm_id},
            ],
        ),
        (
            "https://card.wb.ru/cards/v1/detail",
            [
                {"appType": 1, "curr": "rub", "dest": -1257786, "spp": 30, "nm": nm_id},
                {"nm": nm_id},
            ],
        ),
    ]

    last_err: Optional[Exception] = None
    saw_404 = False
    for url, param_sets in attempts:
        for params in param_sets:
            try:
                return req_json(s, url, params=params, tries=2, sleep_base=0.25, timeout=(4.0, 15.0), debug=debug)
            except Exception as e:
                last_err = e
                # Heuristic: if every endpoint returns 404, the item is likely removed/hidden.
                if " 404 " in f" {e} ":
                    saw_404 = True

    if saw_404:
        raise RuntimeError(
            f"WB карточка не найдена (nmId={nm_id}). Все варианты card.wb.ru вернули 404 — возможно товар удалён/скрыт."
        )
    raise RuntimeError(f"Failed to fetch WB card via API for nmId={nm_id}: {last_err}")


def wb_fetch_card_json(s: requests.Session, nm_id: int, debug: bool = False) -> Dict[str, Any]:
    vol, part = wb_nm_to_vol_part(nm_id)
    url_path = f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"

    # 1) normal path via upstreams route-map
    try:
        host = wb_resolve_card_host(s, vol, debug=debug)
        return req_json(s, f"https://{host}{url_path}", tries=3, sleep_base=0.3, timeout=(4.0, 15.0), debug=debug)
    except Exception as e:
        if debug:
            eprint(f"[wb] resolve+fetch card.json failed: {e}")

    # 2) defensive path: try all known basket hosts from upstreams (dedup)
    try:
        ups = wb_get_upstreams(s, debug=debug)
        cand: List[str] = []
        for sect in ("recommend", "origin"):
            sec = ups.get(sect)
            if not isinstance(sec, dict):
                continue
            rm = sec.get("mediabasket_route_map")
            if isinstance(rm, list) and rm and isinstance(rm[0], dict):
                hs = rm[0].get("hosts")
                if isinstance(hs, list):
                    for h in hs:
                        host = h.get("host")
                        if host:
                            cand.append(str(host))
        # stable order, remove duplicates
        seen: set = set()
        cand_uniq = [h for h in cand if not (h in seen or seen.add(h))]
        js = wb_try_fetch_card_json_from_hosts(s, nm_id, vol, part, cand_uniq, debug=debug)
        if js is not None:
            return js
    except Exception as e:
        if debug:
            eprint(f"[wb] all-hosts card.json fallback failed: {e}")

    # 3) final fallback: card.wb.ru endpoints
    return wb_fetch_card_json_via_api(s, nm_id, debug=debug)


def wb_extract_imt_id(card_js: Dict[str, Any]) -> int:
    # card.json often has imtId at the root; card.wb.ru variants may expose it as "root".
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
    # v4/detail format: {"products": [...]} (no "data" wrapper)
    if isinstance(card_js.get("products"), list) and card_js["products"]:
        p0 = card_js["products"][0]
        if isinstance(p0, dict):
            for k in ("imtId", "imt_id", "imt", "root"):
                v = p0.get(k)
                if v is not None and str(v).isdigit():
                    return int(v)
    raise RuntimeError("Failed to extract imtId from card.json")


def wb_parse_feedbacks(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("feedbacks"), list):
            return payload["feedbacks"]
        if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("feedbacks"), list):
            return payload["data"]["feedbacks"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def wb_host_resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
        return True
    except Exception:
        return False


def wb_simplify_review(r: Dict[str, Any]) -> Dict[str, Any]:
    rating = safe_int(r.get("productValuation", r.get("rating")))
    user = ""
    if isinstance(r.get("wbUserDetails"), dict):
        user = (r["wbUserDetails"].get("name") or "").strip()
    return {
        "id": r.get("id"),
        "rating": rating,
        "created": r.get("createdDate", r.get("created")),
        "user": user,
        "text": (r.get("text") or "").strip(),
        "pros": (r.get("pros") or "").strip(),
        "cons": (r.get("cons") or "").strip(),
    }


def fetch_wb_reviews(
    url: str,
    min_len: int,
    threshold_total: int,
    per_rating: int,
    sleep_s: float,
    take: int,
    fb_from: int,
    fb_to: int,
    debug: bool,
) -> List[Dict[str, Any]]:
    nm_id = wb_parse_nm_id(url)
    eprint(f"[wb] nmId={nm_id}")

    s = requests.Session()
    card = wb_fetch_card_json(s, nm_id, debug=debug)
    imt_id = wb_extract_imt_id(card)
    eprint(f"[wb] imtId={imt_id}")

    candidates = [f"feedbacks{i}.wb.ru" for i in range(int(fb_from), int(fb_to) + 1)]
    good_hosts = [h for h in candidates if wb_host_resolves(h)]
    if debug:
        eprint(f"[wb] dns candidates={candidates}")
        eprint(f"[wb] dns resolved={good_hosts}")
    if not good_hosts:
        raise RuntimeError("Ни один feedbacks*.wb.ru не резолвится (DNS/сеть). Попробуй другой DNS/VPN.")

    rows: List[Dict[str, Any]] = []
    buckets: Optional[Dict[int, List[Dict[str, Any]]]] = None
    sampling_mode = False
    seen_ids: set = set()
    seen_fb: set = set()

    skip = 0
    no_progress = 0
    pbar = tqdm(total=None, unit="rev", desc="reviews", disable=False)

    while True:
        payload = None
        last_err: Optional[Exception] = None

        for h in random.sample(good_hosts, k=len(good_hosts)):
            url_fb = f"https://{h}/feedbacks/v1/{imt_id}"
            try:
                payload = req_json(
                    s,
                    url_fb,
                    params={"take": int(take), "skip": int(skip)},
                    tries=2,
                    sleep_base=0.4,
                    timeout=(4.0, 25.0),
                    debug=debug,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e

        if payload is None:
            raise RuntimeError(f"Не смог получить отзывы ни с одного feedback-host. Последняя ошибка: {last_err}")

        fb = wb_parse_feedbacks(payload)
        if not fb:
            break

        added_this = 0
        for r in fb:
            if safe_int(r.get("nmId")) is not None and int(r.get("nmId")) != int(nm_id):
                continue

            row = wb_simplify_review(r)
            rid = row.get("id")
            if rid is not None:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            else:
                fbk = (str(row.get("created") or ""), str(row.get("rating") or ""), (row.get("text") or "")[:120])
                if fbk in seen_fb:
                    continue
                seen_fb.add(fbk)

            if row["rating"] is None:
                continue
            if min_len and combined_len(row) < min_len:
                continue

            if not sampling_mode:
                rows.append(row)
                added_this += 1
                if len(rows) > threshold_total:
                    sampling_mode = True
                    buckets = seed_buckets_from_rows(rows, per_rating)
                    rows = []
            else:
                assert buckets is not None
                rt = int(row["rating"])
                if rt in buckets and len(buckets[rt]) < per_rating:
                    buckets[rt].append(row)
                    added_this += 1

        if debug:
            eprint(f"[wb] skip={skip} fetched={len(fb)} kept={added_this} total={len(rows) if not sampling_mode else sum(len(v) for v in (buckets or {}).values())}")

        if added_this == 0:
            no_progress += 1
        else:
            no_progress = 0

        if no_progress >= 5:
            eprint("[wb] no new reviews after multiple pages; stopping fetch")
            break

        pbar.update(added_this)

        if sampling_mode and buckets is not None and all_buckets_full(buckets, per_rating):
            break

        skip += int(take)
        time.sleep(max(0.0, sleep_s + random.random() * 0.2))

    pbar.close()

    if sampling_mode and buckets is not None:
        out = stratified_result(buckets)
        eprint(f"[wb] sampled={len(out)} (<= {per_rating} per rating)")
        return out

    eprint(f"[wb] collected={len(rows)}")
    return rows


# ============================================================
# Stage3: scoring + trust
# ============================================================

GENERIC = {
    "все хорошо", "всё хорошо", "норм", "нормально", "ок", "okay", "👍", "всё ок", "все ок",
    "хорошо", "отлично", "супер", "как в описании", "пойдет", "пойдёт",
    "хороший товар", "отличный товар", "всем советую", "рекомендую", "спасибо",
}


def combined_text(r: Dict[str, Any]) -> str:
    t = norm_space(r.get("text") or "")
    p = norm_space(r.get("pros") or "")
    c = norm_space(r.get("cons") or "")
    parts = [x for x in [t, f"Плюсы: {p}" if p else "", f"Минусы: {c}" if c else ""] if x]
    return " ".join(parts).strip()


def count_alpha_num(s: str) -> int:
    return sum(1 for ch in s if ch.isalnum())


def caps_ratio(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for ch in letters if ch.isupper())
    return upp / max(1, len(letters))


def punct_ratio(s: str) -> float:
    punct = sum(1 for ch in s if ch in "!?.,:;…")
    return punct / max(1, len(s)) if s else 0.0


def _strip_templates(s: str) -> str:
    s = norm_space(s).lower()
    s = re.sub(r"^(плюсы|минусы|достоинства|недостатки)\s*:\s*", "", s)
    s = re.sub(r"\b(плюсы|минусы|достоинства|недостатки)\s*:\s*", " ", s)
    return norm_space(s)


def is_generic(s: str) -> bool:
    s2 = _strip_templates(s)
    if s2 in GENERIC:
        return True
    words = _WORD.findall(s2)
    if len(words) <= 2 and len(s2) <= 15:
        return True
    if len(words) <= 8 and any(w in {"все", "всё", "всем", "вся"} for w in words) and any(w in {"нет", "не", "никаких"} for w in words):
        return True
    return False


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def len_score(n_chars: int) -> float:
    return min(1.0, math.log1p(n_chars) / math.log1p(350))


def star_agreement(orig_star: Optional[int], pred_soft: float) -> float:
    if orig_star is None:
        return 1.0
    return max(0.0, 1.0 - abs(orig_star - pred_soft) / 4.0)


def consistency(sent_probs: List[float], pred_soft: float) -> float:
    sent_score = float(sent_probs[2] - sent_probs[0])  # pos - neg in [-1,1]
    star_score = float((pred_soft - 3.0) / 2.0)        # [1..5] -> [-1,1]
    delta = abs(sent_score - star_score)
    return max(0.0, 1.0 - delta / 2.0)


def uniqueness_penalty(dup_count: int) -> float:
    return 1.0 / math.sqrt(max(1, dup_count))


def trust_score(
    txt: str,
    orig_star: Optional[int],
    pred_soft: float,
    sent_probs: List[float],
    dup_count: int,
) -> Tuple[float, Dict[str, Any]]:
    n_chars = len(txt)
    lsc = len_score(n_chars)
    agree = star_agreement(orig_star, pred_soft)
    cons = consistency(sent_probs, pred_soft)

    cap = caps_ratio(txt)
    punc = punct_ratio(txt)
    gen = is_generic(txt)
    uniq = uniqueness_penalty(dup_count)

    pen = 1.0
    if gen:
        pen *= 0.75
    if cap > 0.35:
        pen *= 0.85
    if punc > 0.09:
        pen *= 0.85
    pen *= clamp(uniq, 0.2, 1.0)

    base = 0.05 + 0.35 * lsc + 0.30 * cons + 0.35 * agree
    trust = clamp(base * pen, 0.05, 1.0)

    flags = {
        "len_chars": n_chars,
        "len_score": lsc,
        "generic": bool(gen),
        "caps_ratio": cap,
        "punct_ratio": punc,
        "agreement": agree,
        "consistency": cons,
        "dup_count": int(dup_count),
        "uniq_factor": uniq,
    }
    return trust, flags


def load_tokenizer(path: str):
    with _TOKENIZER_LOCK:
        return _cached_tokenizer(path)


def load_model(path: str, device: torch.device):
    device_key = str(device)
    cpu_int8 = _env_true("RS_CPU_INT8", "0")
    with _MODEL_LOCK:
        return _cached_model(path, device_key, cpu_int8)


def batched(xs: List[str], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def infer_probs(model, tok, texts: List[str], device: torch.device, max_len: int, bs: int, desc: str) -> List[List[float]]:
    out: List[List[float]] = []
    total = (len(texts) + bs - 1) // max(1, bs)
    with torch.inference_mode():
        for chunk in tqdm(batched(texts, bs), total=total, desc=desc, unit="batch"):
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                return_token_type_ids=False,
            )
            if device.type != "cpu":
                enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            out.extend(probs)
    return out


def stage3_build_bundle(
    url: str,
    out_dir: Path,
    sent_model_dir: str,
    rate_model_dir: str,
    min_len_fetch: int = 15,
    threshold: int = 1000,
    per_rating: int = 100,
    sleep_s: float = 0.25,
    wb_take: int = 300,
    fb_from: int = 1,
    fb_to: int = 2,
    min_len: int = 20,
    min_alpha: int = 10,
    batch: int = 64,
    max_len: int = 256,
    device_str: str = "cuda",
    topk: int = 8,
    suspicious_thr: float = 0.30,
    debug: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = url.strip()
    service = detect_service(url)

    # Allow env overrides even if the caller passes fixed args (handy on CPU PaaS)
    # INFER_BATCH / INFER_MAX_LEN are also used by backend/main.py in the full zip.
    batch = int(os.getenv("INFER_BATCH", os.getenv("RS_INFER_BATCH", str(batch))))
    max_len = int(os.getenv("INFER_MAX_LEN", os.getenv("RS_INFER_MAX_LEN", str(max_len))))
    device_str = os.getenv("INFER_DEVICE", os.getenv("RS_INFER_DEVICE", device_str))

    # 1) fetch
    raw_path = out_dir / "reviews_raw.jsonl"
    if service == "wb":
        rows = fetch_wb_reviews(
            url=url,
            min_len=min_len_fetch,
            threshold_total=threshold,
            per_rating=per_rating,
            sleep_s=sleep_s,
            take=wb_take,
            fb_from=fb_from,
            fb_to=fb_to,
            debug=debug,
        )
    else:
        rows = fetch_onliner_reviews(
            url=url,
            min_len=min_len_fetch,
            threshold_total=threshold,
            per_rating=per_rating,
            sleep_s=max(sleep_s, 0.4),
            debug=debug,
        )
    write_jsonl(raw_path, rows)

    # 2) filter + dedupe hashes
    kept_meta: List[Dict[str, Any]] = []
    texts: List[str] = []
    hashes: List[str] = []

    for r in rows:
        txt = combined_text(r)
        if min_len and len(txt) < min_len:
            continue
        if min_alpha and count_alpha_num(txt) < min_alpha:
            continue
        kept_meta.append(r)
        texts.append(txt)
        hashes.append(md5(txt.lower()))

    if not texts:
        raise RuntimeError("После фильтров Stage3 не осталось отзывов. Ослабь --min_len/--min_alpha.")

    # Optional cap to keep CPU inference time bounded on large products
    # Example: RS_MAX_REVIEWS_MODEL=600
    try:
        max_n = int(os.getenv("RS_MAX_REVIEWS_MODEL", "0") or "0")
    except Exception:
        max_n = 0
    if max_n and len(texts) > max_n:
        # Stratified sample by original rating when available
        buckets: Dict[int, List[int]] = {}
        for i, r in enumerate(kept_meta):
            rt = safe_int(r.get("rating"))
            key = int(rt) if rt in (1, 2, 3, 4, 5) else 0
            buckets.setdefault(key, []).append(i)
        all_idx = list(range(len(texts)))
        sampled: List[int] = []
        for key, inds in buckets.items():
            share = len(inds) / max(1, len(all_idx))
            k = max(1, int(round(share * max_n)))
            if len(inds) <= k:
                sampled.extend(inds)
            else:
                sampled.extend(random.sample(inds, k))
        # Final trim (keep order stable)
        sampled = sorted(set(sampled))
        if len(sampled) > max_n:
            sampled = sorted(random.sample(sampled, max_n))
        kept_meta = [kept_meta[i] for i in sampled]
        texts = [texts[i] for i in sampled]
        hashes = [hashes[i] for i in sampled]
        eprint(f"[stage3] RS_MAX_REVIEWS_MODEL cap: using {len(texts)} reviews (of original {len(all_idx)})")

    dup_counts: Dict[str, int] = {}
    for h in hashes:
        dup_counts[h] = dup_counts.get(h, 0) + 1

    # 3) models
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    eprint(f"[device] {device}")

    if device.type == "cpu":
        # Avoid thread thrashing on small CPU instances (Railway/Render/etc.)
        try:
            n = int(os.getenv("RS_TORCH_THREADS", "2"))
            torch.set_num_threads(max(1, n))
            torch.set_num_interop_threads(max(1, min(2, n)))
        except Exception:
            pass

    eprint("[stage3] loading sentiment tokenizer...")
    sent_tok = load_tokenizer(sent_model_dir)
    eprint("[stage3] loading sentiment model...")
    sent_model = load_model(sent_model_dir, device)

    eprint("[stage3] loading rating tokenizer...")
    rate_tok = load_tokenizer(rate_model_dir)
    eprint("[stage3] loading rating model...")
    rate_model = load_model(rate_model_dir, device)

    sent_probs = infer_probs(sent_model, sent_tok, texts, device, max_len, batch, desc="sentiment")  # 3
    rate_probs = infer_probs(rate_model, rate_tok, texts, device, max_len, batch, desc="rating")  # 5

    if sent_probs and len(sent_probs[0]) != 3:
        eprint(f"[warn] sentiment head size={len(sent_probs[0])}, expected 3 (neg/neu/pos)")
    if rate_probs and len(rate_probs[0]) != 5:
        eprint(f"[warn] rating head size={len(rate_probs[0])}, expected 5 (1..5)")

    scored: List[Dict[str, Any]] = []
    for r, txt, hp, ps, pr in zip(kept_meta, texts, hashes, sent_probs, rate_probs):
        soft_star = float(sum((i + 1) * pr[i] for i in range(min(5, len(pr)))))
        hard_star = int(1 + max(range(min(5, len(pr))), key=lambda i: pr[i]))
        orig = safe_int(r.get("rating"))
        trust, flags = trust_score(txt, orig, soft_star, ps, dup_counts[hp])

        row = {
            "id": r.get("id"),
            "created": r.get("created"),
            "user": r.get("user"),
            "source_service": service,
            "orig_star": orig,
            "pred_star_soft": soft_star,
            "pred_star_hard": hard_star,
            "trust": trust,
            "sent_neg": float(ps[0]) if len(ps) > 0 else 0.0,
            "sent_neu": float(ps[1]) if len(ps) > 1 else 0.0,
            "sent_pos": float(ps[2]) if len(ps) > 2 else 0.0,
            "p1": float(pr[0]) if len(pr) > 0 else 0.0,
            "p2": float(pr[1]) if len(pr) > 1 else 0.0,
            "p3": float(pr[2]) if len(pr) > 2 else 0.0,
            "p4": float(pr[3]) if len(pr) > 3 else 0.0,
            "p5": float(pr[4]) if len(pr) > 4 else 0.0,
            "text": txt,
            "flags": flags,
        }
        scored.append(row)

    df = pd.DataFrame(scored)
    w = df["trust"].to_numpy(dtype=float)
    pred_soft = df["pred_star_soft"].to_numpy(dtype=float)

    truth_stars = float((w * pred_soft).sum() / max(1e-9, w.sum()))
    avg_orig = float(df["orig_star"].dropna().mean()) if df["orig_star"].notna().any() else float("nan")
    suspicious_share = float((df["trust"] < float(suspicious_thr)).mean())
    polarity_share = float(((df["orig_star"] == 1) | (df["orig_star"] == 5)).mean()) if df["orig_star"].notna().any() else float("nan")

    sent_mix = {
        "neg": float((w * df["sent_neg"].to_numpy()).sum() / max(1e-9, w.sum())),
        "neu": float((w * df["sent_neu"].to_numpy()).sum() / max(1e-9, w.sum())),
        "pos": float((w * df["sent_pos"].to_numpy()).sum() / max(1e-9, w.sum())),
    }

    def top_k_rows(rows: List[Dict[str, Any]], key: str, k: int, reverse: bool) -> List[Dict[str, Any]]:
        rs = sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:k]
        out = []
        for r in rs:
            out.append({
                "id": r.get("id"),
                "created": r.get("created"),
                "user": r.get("user"),
                "orig_star": r.get("orig_star"),
                "pred_star_soft": r.get("pred_star_soft"),
                "pred_star_hard": r.get("pred_star_hard"),
                "trust": r.get("trust"),
                "text": r.get("text"),
            })
        return out

    trusted = top_k_rows(scored, "trust", topk, reverse=True)
    suspicious = top_k_rows(scored, "trust", topk, reverse=False)

    ranked = sorted(scored, key=lambda r: float(r["trust"]), reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["trust_percentile"] = float(1.0 - (i - 1) / max(1, len(ranked) - 1))

    summary = {
        "input": {
            "url": url,
            "service": service,
            "fetch": {
                "min_len_fetch": min_len_fetch,
                "threshold": threshold,
                "per_rating": per_rating,
                "sleep": sleep_s,
                "wb_take": wb_take,
                "fb_from": fb_from,
                "fb_to": fb_to,
            },
            "filter": {"min_len": min_len, "min_alpha": min_alpha},
            "models": {"sent_model": sent_model_dir, "rate_model": rate_model_dir},
        },
        "counts": {"raw": int(len(rows)), "kept": int(len(df))},
        "stars": {"avg_orig": avg_orig, "truth_stars": truth_stars, "polarity_share": polarity_share},
        "trust": {
            "suspicious_share": suspicious_share,
            "threshold": float(suspicious_thr),
            "trust_mean": float(df["trust"].mean()),
            "trust_median": float(df["trust"].median()),
        },
        "sentiment_weighted": sent_mix,
        "top": {"trusted": trusted, "suspicious": suspicious},
        "artifacts": {
            "reviews_raw_jsonl": str(raw_path.name),
            "reviews_scored_csv": "reviews_scored.csv",
            "reviews_ranked_jsonl": "reviews_ranked.jsonl",
        },
    }

    # Save scored CSV (without flags)
    scored_csv = out_dir / "reviews_scored.csv"
    df.drop(columns=["flags"], errors="ignore").to_csv(scored_csv, index=False, encoding="utf-8")

    # Save ranked JSONL
    ranked_jsonl = out_dir / "reviews_ranked.jsonl"
    write_jsonl(ranked_jsonl, ranked)

    # Save bundle
    bundle_path = out_dir / "stage3_bundle.json"
    write_json(bundle_path, {"summary": summary, "reviews": ranked})

    eprint(f"[ok] raw={len(rows)} kept={len(df)} truth_stars={truth_stars:.3f} suspicious_share={suspicious_share:.3f}")
    eprint(f"[saved] {bundle_path}")
    return bundle_path


# ============================================================
# Stage4: embeddings + RAG + summary
# ============================================================

@dataclass
class EmbeddingProviderConfig:
    provider: str = "openai"  # openai
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 96
    timeout_s: int = 60
    max_retries: int = 6


def embed_texts_openai(texts: List[str], cfg: EmbeddingProviderConfig) -> np.ndarray:
    api_key = os.getenv(cfg.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing {cfg.api_key_env}. Set it in your shell.")
    url = cfg.base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    out: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), cfg.batch_size), desc="embeddings", unit="batch"):
        batch = texts[i:i + cfg.batch_size]
        payload = {"model": cfg.model, "input": batch}
        retry = 0
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                items = data["data"]
                items.sort(key=lambda x: x["index"])
                vecs = [np.array(it["embedding"], dtype=np.float32) for it in items]
                out.append(np.stack(vecs, axis=0))
                break
            except Exception as e:
                retry += 1
                if retry > cfg.max_retries:
                    raise
                sleep_s = min(10.0, 0.5 * (2 ** (retry - 1)))
                eprint(f"[embed retry {retry}/{cfg.max_retries}] {e} -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)

    mat = np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    return mat


def embed_texts(texts: List[str], cfg: EmbeddingProviderConfig) -> np.ndarray:
    if cfg.provider != "openai":
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")
    return embed_texts_openai(texts, cfg)


@dataclass
class LLMProviderConfig:
    provider: str = "openai"  # openai | groq
    model: str = "gpt-4o-mini"
    timeout_s: int = 90
    max_retries: int = 6
    temperature: float = 0.2


def llm_chat_openai_compatible(messages: List[Dict[str, str]], base_url: str, api_key_env: str, cfg: LLMProviderConfig) -> str:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}. Set it in your shell.")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": cfg.model, "messages": messages, "temperature": cfg.temperature}

    retry = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            retry += 1
            if retry > cfg.max_retries:
                raise
            sleep_s = min(10.0, 0.5 * (2 ** (retry - 1)))
            eprint(f"[llm retry {retry}/{cfg.max_retries}] {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)


def llm_generate(messages: List[Dict[str, str]], cfg: LLMProviderConfig) -> str:
    if cfg.provider == "openai":
        return llm_chat_openai_compatible(messages, "https://api.openai.com/v1", "OPENAI_API_KEY", cfg)
    if cfg.provider == "groq":
        return llm_chat_openai_compatible(messages, "https://api.groq.com/openai/v1", "GROQ_API_KEY", cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


@dataclass
class IndexConfig:
    space: str = "cosine"
    ef_construction: int = 200
    M: int = 48
    ef_search: int = 128


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _normalize_text(s: str) -> str:
    return norm_space(_safe_str(s))


def _fingerprint_review(r: Dict[str, Any]) -> str:
    payload = f"{_safe_str(r.get('id'))}||{_normalize_text(r.get('text'))}||{_safe_str(r.get('created'))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_stage3_bundle(bundle_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bundle = read_json(bundle_path)
    summary = bundle.get("summary", {}) if isinstance(bundle, dict) else {}
    reviews = bundle.get("reviews", []) if isinstance(bundle, dict) else []
    if not isinstance(reviews, list):
        raise ValueError("stage3_bundle.json: bundle['reviews'] must be a list")

    uniq: Dict[str, Dict[str, Any]] = {}
    for r in reviews:
        if not isinstance(r, dict):
            continue
        key = _fingerprint_review(r)
        if key not in uniq:
            rr = dict(r)
            rr["id"] = _safe_str(rr.get("id"))
            rr["text"] = _normalize_text(rr.get("text"))
            rr["source_service"] = _safe_str(rr.get("source_service") or rr.get("service") or rr.get("source") or "")
            uniq[key] = rr

    out = list(uniq.values())
    out.sort(key=lambda x: float(x.get("trust", 0.0) or 0.0), reverse=True)
    return summary, out


def suspicious_share_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    for path in (("trust", "suspicious_share"), ("truth", "suspicious_share")):
        cur: Any = summary
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            try:
                return float(cur)
            except Exception:
                return None
    return None


def build_index(emb: np.ndarray, cfg: IndexConfig) -> Any:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed. Install: pip install hnswlib")
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("Empty embeddings matrix")
    dim = int(emb.shape[1])
    idx = hnswlib.Index(space=cfg.space, dim=dim)
    idx.init_index(max_elements=int(emb.shape[0]), ef_construction=cfg.ef_construction, M=cfg.M)
    idx.add_items(emb, np.arange(emb.shape[0], dtype=np.int64))
    idx.set_ef(cfg.ef_search)
    return idx


def save_index(idx: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save_index(str(out_path))


def load_index(index_path: Path, dim: int, cfg: IndexConfig) -> Any:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed. Install: pip install hnswlib")
    idx = hnswlib.Index(space=cfg.space, dim=dim)
    idx.load_index(str(index_path))
    idx.set_ef(cfg.ef_search)
    return idx


def trust_val(r: Dict[str, Any]) -> float:
    try:
        return float(r.get("trust", 0.0) or 0.0)
    except Exception:
        return 0.0


def len_score_val(r: Dict[str, Any]) -> float:
    flags = r.get("flags", {}) if isinstance(r.get("flags", {}), dict) else {}
    try:
        return float(flags.get("len_score", 1.0) or 1.0)
    except Exception:
        return 1.0


def rerank(sim: np.ndarray, cand_reviews: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    scored = []
    for s, r in zip(sim.tolist(), cand_reviews):
        t = trust_val(r)
        ls = len_score_val(r)
        w = (0.70 + 0.30 * clamp(t, 0.0, 1.0)) * (0.80 + 0.20 * clamp(ls, 0.0, 1.0))
        scored.append((float(s) * w, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve(query: str, idx: Any, meta: List[Dict[str, Any]], emb_cfg: EmbeddingProviderConfig,
             top_k: int = 30, rerank_k: int = 12) -> List[Tuple[float, Dict[str, Any]]]:
    if not meta:
        return []
    k = min(int(top_k), len(meta))
    if k <= 0:
        return []
    qv = embed_texts([query], emb_cfg)[0]
    labels, distances = idx.knn_query(qv, k=k)
    labels = labels[0].tolist()
    dist = distances[0]
    sim = 1.0 - dist
    cand = [meta[i] for i in labels]
    return rerank(sim, cand)[: min(int(rerank_k), len(cand))]


def format_context(items: List[Tuple[float, Dict[str, Any]]], max_chars_each: int = 500) -> List[Dict[str, Any]]:
    ctx = []
    for score, r in items:
        txt = _safe_str(r.get("text", ""))
        if len(txt) > max_chars_each:
            txt = txt[:max_chars_each].rstrip() + "…"
        ctx.append({
            "id": _safe_str(r.get("id")),
            "orig_star": r.get("orig_star"),
            "pred_star_hard": r.get("pred_star_hard"),
            "trust": r.get("trust"),
            "created": r.get("created"),
            "score": round(float(score), 6),
            "text": txt,
        })
    return ctx


def rag_answer(question: str, ctx: List[Dict[str, Any]], llm_cfg: LLMProviderConfig) -> str:
    system = (
        "Ты эксперт по анализу отзывов. Твоя задача — дать чёткий, лаконичный ответ на вопрос пользователя, "
        "опираясь ТОЛЬКО на предоставленные отзывы.\n"
        "\n"
        "ФОРМАТ ОТВЕТА:\n"
        "Начинай сразу с сути, без заголовков и нумерации.\n"
        "\n"
        "Структура (БЕЗ MARKDOWN):\n"
        "• Первый абзац (2-4 предложения): краткий ответ с ключевыми фактами\n"
        "• Детали через абзацы:\n"
        "  - Что хвалят (если есть)\n"
        "  - Что критикуют (если есть)\n"
        "  - Частота упоминаний ('большинство' / 'несколько человек' / 'единичные случаи')\n"
        "• Последний абзац: практический вывод для покупателя\n"
        "\n"
        "ПРАВИЛА:\n"
        "✓ Пиши простым языком, без звёздочек, заголовков, маркеров\n"
        "✓ Используй абзацы для разделения мыслей\n"
        "✓ Приводи конкретные цитаты в кавычках, но БЕЗ упоминания ID\n"
        "✓ Указывай количество: 'почти все', '7 из 10', 'несколько', 'один покупатель'\n"
        "✓ Будь объективен: покажи и плюсы, и минусы\n"
        "✓ Если противоречия — так и скажи: 'мнения разделились'\n"
        "\n"
        "✗ НЕ выдумывай факты\n"
        "✗ НЕ используй markdown (**, ##, -)\n"
        "✗ НЕ пиши 'Краткий вывод:', 'Детальный анализ:' и т.п.\n"
        "✗ НЕ упоминай ID, trust score, технические детали\n"
        "\n"
        "Если в отзывах нет ответа — честно скажи об этом.\n"
    )
    user = f"Вопрос: {question}\n\nОтзывы (JSON):\n{json.dumps(ctx, ensure_ascii=False)}"
    return llm_generate([{"role": "system", "content": system}, {"role": "user", "content": user}], llm_cfg)


def summarize_product(summary_obj: Dict[str, Any], reviews: List[Dict[str, Any]], llm_cfg: LLMProviderConfig,
                      max_evidence: int = 90) -> Dict[str, Any]:
    def pred_star(r: Dict[str, Any]) -> int:
        v = r.get("pred_star_hard", None)
        if v is None:
            v = r.get("orig_star", None)
        try:
            return int(v)
        except Exception:
            return 0

    rs = sorted(reviews, key=lambda r: (trust_val(r), pred_star(r)), reverse=True)
    pos = [r for r in rs if pred_star(r) >= 4]
    neg = [r for r in rs if 1 <= pred_star(r) <= 2]
    mid = [r for r in rs if pred_star(r) == 3]

    ev: List[Dict[str, Any]] = []
    ev.extend(pos[: max_evidence // 2])
    ev.extend(neg[: max_evidence // 3])
    ev.extend(mid[: max(0, max_evidence - len(ev))])
    ev = ev[:max_evidence]

    ctx = format_context([(1.0, r) for r in ev], max_chars_each=650)

    susp = suspicious_share_from_summary(summary_obj)
    truth_stars = None
    try:
        truth_stars = float(((summary_obj.get("stars") or {}).get("truth_stars")))
    except Exception:
        truth_stars = None

    meta = {
        "source_service": _safe_str((summary_obj.get("input", {}) or {}).get("service")) if isinstance(summary_obj.get("input"), dict) else _safe_str(summary_obj.get("service")),
        "url": _safe_str((summary_obj.get("input", {}) or {}).get("url")) if isinstance(summary_obj.get("input"), dict) else _safe_str(summary_obj.get("url")),
        "counts": summary_obj.get("counts"),
        "truth_stars": truth_stars,
        "suspicious_share": susp,
    }

    system = (
        "Ты делаешь витринное резюме по отзывам для карточки товара.\n"
        "Верни СТРОГО JSON (без текста вокруг) по схеме:\n"
        "{\n"
        "  \"one_liner\": str,\n"
        "  \"score\": {\"value\": float, \"scale\": 5, \"note\": str},\n"
        "  \"pros\": [{\"point\": str, \"evidence_ids\": [str]}],\n"
        "  \"cons\": [{\"point\": str, \"evidence_ids\": [str]}],\n"
        "  \"aspects\": [{\"aspect\": str, \"summary\": str, \"mentions\": int, \"evidence_ids\": [str]}],\n"
        "  \"fit_size\": {\"summary\": str, \"evidence_ids\": [str]} | null,\n"
        "  \"quality_issues\": [{\"issue\": str, \"evidence_ids\": [str]}],\n"
        "  \"who_it_is_for\": [str],\n"
        "  \"who_should_avoid\": [str],\n"
        "  \"confidence\": {\"level\": \"low\"|\"medium\"|\"high\", \"why\": str},\n"
        "  \"notes\": {\"suspicious_share\": float|null, \"truth_stars\": float|null, \"data\": {...}}\n"
        "}\n"
        "Правила:\n"
        "- Каждый пункт в pros/cons/quality_issues/aspects ДОЛЖЕН иметь 2-6 evidence_ids.\n"
        "- В aspects используй такие темы: размер/посадка, ткань/толщина, упаковка, внешний вид/цвет, брак/швы, комфорт.\n"
        "- score.value старайся привязать к truth_stars (если он есть), иначе делай осторожно.\n"
        "- Нельзя придумывать факты (состав, брендовые обещания), если их нет в тексте.\n"
    )

    user = f"Метаданные (JSON): {json.dumps(meta, ensure_ascii=False)}\n\nОтзывы (JSON):\n{json.dumps(ctx, ensure_ascii=False)}"
    raw = llm_generate([{"role": "system", "content": system}, {"role": "user", "content": user}], llm_cfg)

    try:
        out = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise RuntimeError("LLM не вернул JSON. Raw:\n" + raw[:800])
        out = json.loads(m.group(0))

    out.setdefault("notes", {})
    out["notes"]["suspicious_share"] = susp
    out["notes"]["truth_stars"] = truth_stars
    out["notes"]["data"] = meta
    return out


def rag_build_from_bundle(bundle_path: Path, rag_dir: Path, emb_model: str, emb_batch: int) -> None:
    rag_dir.mkdir(parents=True, exist_ok=True)
    summary, reviews = load_stage3_bundle(bundle_path)
    if not reviews:
        raise RuntimeError("No reviews in bundle")

    texts = [r["text"] for r in reviews]
    emb_cfg = EmbeddingProviderConfig(model=emb_model, batch_size=emb_batch)
    embs = embed_texts(texts, emb_cfg)

    idx = build_index(embs, IndexConfig())
    save_index(idx, rag_dir / "hnsw.index")

    meta_rows = []
    for i, r in enumerate(reviews):
        rr = dict(r)
        rr["_row"] = i
        meta_rows.append(rr)

    write_json(rag_dir / "meta.json", {"dim": int(embs.shape[1]), "count": int(embs.shape[0]), "bundle": str(bundle_path)})
    write_jsonl(rag_dir / "meta.jsonl", meta_rows)

    eprint(f"OK: built index with {len(meta_rows)} docs, dim={embs.shape[1]} -> {rag_dir}")


def load_meta(meta_jsonl: Path) -> List[Dict[str, Any]]:
    rows = []
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x.get("_row", 0)))
    return rows


def ask_with_rag(rag_dir: Path, question: str, emb_model: str, emb_batch: int,
                 llm_provider: str, llm_model: str, temp: float,
                 top_k: int, rerank_k: int, max_chars_each: int) -> Tuple[str, List[Dict[str, Any]]]:
    meta_info = read_json(rag_dir / "meta.json")
    dim = int(meta_info["dim"])
    meta = load_meta(rag_dir / "meta.jsonl")
    idx = load_index(rag_dir / "hnsw.index", dim=dim, cfg=IndexConfig())

    emb_cfg = EmbeddingProviderConfig(model=emb_model, batch_size=emb_batch)
    llm_cfg = LLMProviderConfig(provider=llm_provider, model=llm_model, temperature=temp)

    ranked = retrieve(question, idx, meta, emb_cfg, top_k=top_k, rerank_k=rerank_k)
    ctx = format_context(ranked, max_chars_each=max_chars_each)
    answer = rag_answer(question, ctx, llm_cfg)
    return answer, ctx


# ============================================================
# CLI (ONE TOOL)
# ============================================================

def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    bundle_path = stage3_build_bundle(
        url=args.url,
        out_dir=out_dir,
        sent_model_dir=args.sent_model,
        rate_model_dir=args.rate_model,
        min_len_fetch=args.min_len_fetch,
        threshold=args.threshold,
        per_rating=args.per_rating,
        sleep_s=args.sleep,
        wb_take=args.wb_take,
        fb_from=args.fb_from,
        fb_to=args.fb_to,
        min_len=args.min_len,
        min_alpha=args.min_alpha,
        batch=args.batch,
        max_len=args.max_len,
        device_str=args.device,
        topk=args.topk,
        suspicious_thr=args.suspicious_thr,
        debug=args.debug,
    )

    rag_dir = Path(args.rag_dir) if args.rag_dir else (out_dir / "rag")
    rag_build_from_bundle(bundle_path, rag_dir, emb_model=args.emb_model, emb_batch=args.emb_batch)

    # Summary
    if args.make_summary:
        summary_obj, reviews = load_stage3_bundle(bundle_path)
        llm_cfg = LLMProviderConfig(provider=args.llm_provider, model=args.llm_model, temperature=args.temp)
        out = summarize_product(summary_obj, reviews, llm_cfg, max_evidence=args.max_evidence)
        out_path = Path(args.summary_out) if args.summary_out else (out_dir / "stage4_summary.json")
        write_json(out_path, out)
        eprint(f"OK: summary -> {out_path}")

    # Optional questions
    if args.question:
        ans, ctx = ask_with_rag(
            rag_dir=rag_dir,
            question=args.question,
            emb_model=args.emb_model,
            emb_batch=args.emb_batch,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            temp=args.temp,
            top_k=args.top_k,
            rerank_k=args.rerank_k,
            max_chars_each=args.max_chars_each,
        )
        print(ans)
        if args.save_answer_json:
            write_json(Path(args.save_answer_json), {"question": args.question, "answer": ans, "context": ctx})


def cmd_stage3(args: argparse.Namespace) -> None:
    stage3_build_bundle(
        url=args.url,
        out_dir=Path(args.out_dir),
        sent_model_dir=args.sent_model,
        rate_model_dir=args.rate_model,
        min_len_fetch=args.min_len_fetch,
        threshold=args.threshold,
        per_rating=args.per_rating,
        sleep_s=args.sleep,
        wb_take=args.wb_take,
        fb_from=args.fb_from,
        fb_to=args.fb_to,
        min_len=args.min_len,
        min_alpha=args.min_alpha,
        batch=args.batch,
        max_len=args.max_len,
        device_str=args.device,
        topk=args.topk,
        suspicious_thr=args.suspicious_thr,
        debug=args.debug,
    )


def cmd_rag_build(args: argparse.Namespace) -> None:
    rag_build_from_bundle(Path(args.bundle), Path(args.rag_dir), emb_model=args.emb_model, emb_batch=args.emb_batch)


def cmd_ask(args: argparse.Namespace) -> None:
    ans, ctx = ask_with_rag(
        rag_dir=Path(args.rag_dir),
        question=args.question,
        emb_model=args.emb_model,
        emb_batch=args.emb_batch,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temp=args.temp,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        max_chars_each=args.max_chars_each,
    )
    print(ans)
    if args.save_json:
        write_json(Path(args.save_json), {"question": args.question, "answer": ans, "context": ctx})


def cmd_summarize(args: argparse.Namespace) -> None:
    summary_obj, reviews = load_stage3_bundle(Path(args.bundle))
    llm_cfg = LLMProviderConfig(provider=args.llm_provider, model=args.llm_model, temperature=args.temp)
    out = summarize_product(summary_obj, reviews, llm_cfg, max_evidence=args.max_evidence)
    write_json(Path(args.out), out)
    eprint(f"OK: summary -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reviewscope_all.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared stage3 args builder
    def add_stage3_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--url", required=True, help="WB URL/nmId или Onliner URL/key")
        pp.add_argument("--out_dir", default="stage3_out", help="Куда сохранять stage3 артефакты")
        pp.add_argument("--sent_model", required=True, help="Путь к sentiment модели (HF dir)")
        pp.add_argument("--rate_model", required=True, help="Путь к rating модели (HF dir)")
        pp.add_argument("--min_len_fetch", type=int, default=30)
        pp.add_argument("--threshold", type=int, default=1000)
        pp.add_argument("--per_rating", type=int, default=100)
        pp.add_argument("--sleep", type=float, default=0.25)
        pp.add_argument("--wb_take", type=int, default=300)
        # IMPORTANT: only 1..2 are working for you
        pp.add_argument("--fb_from", type=int, default=1, help="WB: feedbacks host range start (WORKSHOW: 1)")
        pp.add_argument("--fb_to", type=int, default=2, help="WB: feedbacks host range end (WORKSHOW: 2)")
        pp.add_argument("--min_len", type=int, default=40)
        pp.add_argument("--min_alpha", type=int, default=20)
        pp.add_argument("--batch", type=int, default=64)
        pp.add_argument("--max_len", type=int, default=256)
        pp.add_argument("--device", default="cuda")
        pp.add_argument("--topk", type=int, default=8)
        pp.add_argument("--suspicious_thr", type=float, default=0.30)
        pp.add_argument("--debug", action="store_true")

    # RUN (full pipeline)
    run = sub.add_parser("run", help="Full pipeline: Stage3 -> RAG build -> optional summary + optional ask")
    add_stage3_args(run)
    run.add_argument("--rag_dir", default="", help="Куда сохранить индекс (по умолчанию out_dir/rag)")
    run.add_argument("--emb_model", default="text-embedding-3-small")
    run.add_argument("--emb_batch", type=int, default=96)
    run.add_argument("--llm_provider", default="openai", choices=["openai", "groq"])
    run.add_argument("--llm_model", default="gpt-4o-mini")
    run.add_argument("--temp", type=float, default=0.2)
    run.add_argument("--make_summary", action="store_true", help="Сделать stage4_summary.json")
    run.add_argument("--summary_out", default="", help="Куда сохранить summary (по умолчанию out_dir/stage4_summary.json)")
    run.add_argument("--max_evidence", type=int, default=90)
    run.add_argument("--question", default="", help="Опционально: сразу задать вопрос после build")
    run.add_argument("--top_k", type=int, default=30)
    run.add_argument("--rerank_k", type=int, default=12)
    run.add_argument("--max_chars_each", type=int, default=500)
    run.add_argument("--save_answer_json", default="", help="Сохранить ответ+контекст в JSON")
    run.set_defaults(func=cmd_run)

    # Stage3 only
    st3 = sub.add_parser("stage3", help="Only Stage3: fetch + scoring + stage3_bundle.json")
    add_stage3_args(st3)
    st3.set_defaults(func=cmd_stage3)

    # RAG build
    rb = sub.add_parser("rag_build", help="Build RAG index from stage3_bundle.json")
    rb.add_argument("--bundle", required=True)
    rb.add_argument("--rag_dir", default="stage3_out/rag")
    rb.add_argument("--emb_model", default="text-embedding-3-small")
    rb.add_argument("--emb_batch", type=int, default=96)
    rb.set_defaults(func=cmd_rag_build)

    # Ask
    ask = sub.add_parser("ask", help="Ask a question via RAG")
    ask.add_argument("--rag_dir", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--emb_model", default="text-embedding-3-small")
    ask.add_argument("--emb_batch", type=int, default=96)
    ask.add_argument("--llm_provider", default="openai", choices=["openai", "groq"])
    ask.add_argument("--llm_model", default="gpt-4o-mini")
    ask.add_argument("--temp", type=float, default=0.2)
    ask.add_argument("--top_k", type=int, default=30)
    ask.add_argument("--rerank_k", type=int, default=12)
    ask.add_argument("--max_chars_each", type=int, default=500)
    ask.add_argument("--save_json", default="")
    ask.set_defaults(func=cmd_ask)

    # Summarize
    sm = sub.add_parser("summarize", help="Generate stage4 summary JSON with evidence ids")
    sm.add_argument("--bundle", required=True)
    sm.add_argument("--out", default="stage4_summary.json")
    sm.add_argument("--max_evidence", type=int, default=90)
    sm.add_argument("--llm_provider", default="openai", choices=["openai", "groq"])
    sm.add_argument("--llm_model", default="gpt-4o-mini")
    sm.add_argument("--temp", type=float, default=0.2)
    sm.set_defaults(func=cmd_summarize)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
