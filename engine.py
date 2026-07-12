#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py — a generic, multi-tenant AI receptionist engine.

One engine, many offices. The engine code below is the SAME for every office.
Everything office-specific lives in a *profile* (loaded per tenant) and in
*backends* (RAG, LLM, calendar, intent classifier) that the engine calls but
does not contain. Add a new office by dropping in a profile — you should never
have to edit this file to onboard someone.

Request flow (first match wins, top to bottom):
    1. Session state machine   (mid-conversation booking, etc.)
    2. FSM rule match          (office's .jsonl answer rules)
    3. Scheduling regex         (catches "book an appointment" phrasing)
    4. Fragment / clarify gate  (too-short, ambiguous input)
    5. ML intent classify       (optional external service)
    6. RAG retrieval            (optional knowledge base)
    7. LLM generation           (optional general fallback)
    8. Email fallback           (always available)

Run it:
    pip install flask flask-cors requests
    python engine.py
With no configuration it still runs — backends simply stay disabled and the
engine answers from rules + fallback (graceful degradation).
"""

from __future__ import annotations

import os
import re
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS


# =============================================================================
# SECTION 1 — Text normalization & parsing helpers
# These are pure functions with no office-specific state. They belong to the
# engine and are shared by every tenant.
# =============================================================================

# ASR ("mis-hear") fixes for voice deployments. Generic ones live here; office
# specific ones (like a provider's name) can be added via the profile.
_ASR_ALIASES: List[Tuple[str, str]] = [
    (r"\bdoctor\b", "dr"),
    (r"\bdr\.\b", "dr"),
]

# Query expansions so short questions match rules more reliably.
_SYNONYM_EXPANSIONS: List[Tuple[str, str]] = [
    (r"\binsurance\b", "insurance coverage plan"),
    (r"\bhours\b", "hours office hours open closed"),
    (r"\blocation\b", "location address directions"),
]


def normalize_text(s: str, extra_aliases: Optional[List[Tuple[str, str]]] = None) -> str:
    """Lowercase, strip punctuation, apply ASR fixes and synonym expansions."""
    s = (s or "").lower().strip()
    s = re.sub(r"[\u2019\u2018]", "'", s)  # smart quotes -> '
    for pat, repl in _ASR_ALIASES:
        s = re.sub(pat, repl, s, flags=re.I)
    for pat, repl in (extra_aliases or []):
        s = re.sub(pat, repl, s, flags=re.I)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in _SYNONYM_EXPANSIONS:
        s = re.sub(pat, repl, s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


_WHAT_ABOUT_RE = re.compile(r"^what\s+about\b", re.I)
_TOO_SHORT_RE = re.compile(r"^[a-z]{1,3}$")
_MEANINGFUL_RE = re.compile(
    r"\b(training|education|residency|fellowship|background|insurance|coverage|"
    r"plan|hours|open|closed|location|address|directions|hospital|affiliation|"
    r"appointment|appt|schedule|book|consult|results|billing|available|"
    r"availability|openings)\b",
    re.I,
)


def is_fragment(text: str) -> bool:
    """True if input is too short/ambiguous to route confidently."""
    raw = (text or "").strip()
    if not raw:
        return True
    if _WHAT_ABOUT_RE.match(raw):
        return False
    s = normalize_text(raw)
    if _MEANINGFUL_RE.search(s):
        return False
    if _TOO_SHORT_RE.match(s):
        return True
    if len(s.split()) <= 3 and "?" not in raw:
        return True
    return False


_SMALLTALK_RE = re.compile(
    r"\b(joke|funny|laugh|riddle|story|chat|talk|how are you|what's up|whats up)\b",
    re.I,
)

# Ways a caller signals they want OUT of the booking flow.
CANCEL_RE = re.compile(
    r"\b(cancel|never\s*mind|nevermind|forget it|forget that|stop|start over|"
    r"different question|another question|something else|other question|"
    r"not now|no thanks|no thank you|quit|exit|go back|hold on|wait)\b",
    re.I,
)

# How many times we re-ask a scheduling step before letting the caller go.
MAX_REASKS = 2

# Minimum retrieval score before we'll let the LLM answer FROM a KB entry.
#
# This is deliberately MUCH lower than the direct-answer threshold, and the
# reason matters:
#
#   * Answering DIRECTLY means speaking one KB entry verbatim. If the top match
#     is wrong, the caller hears a wrong answer. So that bar is high (~0.60).
#
#   * GROUNDING means handing the top-k entries to the LLM as source material.
#     The model reads all of them and picks what's relevant — so the correct
#     entry only has to be NEAR the top, not exactly first. A small embedding
#     model often ranks the right answer 2nd or 3rd; grounding rescues those.
#
# Measure it for your own KB with:  python tune_rag.py --compare
GROUNDING_MIN_SCORE = float(os.environ.get("GROUNDING_MIN_SCORE", "0.30"))

# Let the LLM answer with NO retrieved facts to work from? Default: NO.
# Tested on a real instruction-following model (Mistral-7B-Instruct-v0.2): asked
# "Do you validate parking?" it fabricated a complete, confident parking-
# validation policy. The office has no such policy. A fluent invention is more
# dangerous than obvious gibberish, because nobody can tell it's wrong.
ALLOW_UNGROUNDED_LLM = os.environ.get("ALLOW_UNGROUNDED_LLM", "0") == "1"

# Text shaped like a question — used so the "name" step can't swallow one.
_QUESTIONY_RE = re.compile(
    r"\?|^\s*(do|does|did|can|could|will|would|is|are|was|were|should|what|"
    r"where|when|why|how|who|which)\b",
    re.I,
)


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK_RE.search(text or ""))


# ---- scheduling: day / time / slot / dob parsing ----------------------------

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

WEEKDAY_MAP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2,
    "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4,
    "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}

_SLOT_PICK_RE = re.compile(
    r"^\s*(?:book\s*)?(\d{1,2})\s*$|"
    r"\b(book|choose|pick)\b.*\b(\d{1,2})\b|"
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\b",
    re.I,
)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)

SCHED_INTENT_RE = re.compile(
    r"\b(schedule|book|make|set\s*up|get)\b.*\b(appointment|appt|visit|consult|consultation)\b"
    r"|\b(appointment|appt|visit|consult|consultation)\b.*\b(schedule|book|make|get)\b"
    r"|\b(soonest|earliest|next)\b.*\b(appointment|appt|opening|availability)\b"
    r"|\b(available|availability|openings?)\b.*\b(dates?|times?)\b"
    r"|\b(dates?|times?)\b.*\b(available|availability|openings?)\b",
    re.I,
)


def _today() -> date:
    return datetime.now().date()


def _fmt_slot(iso: str) -> str:
    """Format an ISO slot start into a readable label, e.g. 'Mon Jul 13, 9:00 AM'."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
    except Exception:
        return iso


def parse_dob_iso(text: str) -> Optional[str]:
    """Parse a date of birth into ISO (YYYY-MM-DD). Returns None if unparseable.

    Built for SPOKEN input: speech-to-text renders "April twenty-second,
    nineteen seventy" in many shapes — "04 22 1970", "4/22/70", "April 22 1970",
    even "04.22.1970". We accept separators of /, -, ., or plain spaces.
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    t = re.sub(r"\b(dob\s+is|date\s+of\s+birth\s+is|born\s+on|born|my|it'?s|is)\b", " ", t)
    t = re.sub(r"[,:]", " ", t)
    t = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    # ISO: 1970-04-22
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass

    # Month name: April 22 1970
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})\s+(\d{2,4})\b", t)
    if m and m.group(1) in MONTHS:
        try:
            mm, dd, yy = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
            if yy < 100:
                yy = 1900 + yy if yy > 30 else 2000 + yy
            return date(yy, mm, dd).isoformat()
        except ValueError:
            pass

    # Numeric with ANY separator — /, -, ., or spaces:
    #   04/22/1970   4-22-70   04.22.1970   "04 22 1970"   "4 22 1970"
    m = re.search(r"\b(\d{1,2})\s*[/\-. ]\s*(\d{1,2})\s*[/\-. ]\s*(\d{2,4})\b", t)
    if m:
        try:
            mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if yy < 100:
                yy = 1900 + yy if yy > 30 else 2000 + yy
            return date(yy, mm, dd).isoformat()
        except ValueError:
            pass

    # Bare 8-digit run: "04221970"
    m = re.search(r"\b(\d{2})(\d{2})(\d{4})\b", t)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat()
        except ValueError:
            pass
    return None


def parse_day_choice(text: str, horizon_days: int = 30) -> Optional[str]:
    """Parse a spoken/typed day ('Monday', 'tomorrow', 'Jan 19') into ISO date."""
    raw = (text or "").strip()
    if not raw:
        return None
    s = normalize_text(raw)

    if "today" in s:
        return _today().isoformat()
    if "tomorrow" in s:
        return (_today() + timedelta(days=1)).isoformat()

    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", raw)
    if m:
        try:
            mm, dd = int(m.group(1)), int(m.group(2))
            yy = _today().year if m.group(3) is None else int(m.group(3))
            if yy < 100:
                yy = 1900 + yy if yy > 30 else 2000 + yy
            return date(yy, mm, dd).isoformat()
        except ValueError:
            return None

    m = re.search(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b", raw, re.I)
    if m and (m.group(1) or "").lower() in MONTHS:
        try:
            mm = MONTHS[m.group(1).lower()]
            dd = int(m.group(2))
            yy = int(m.group(3)) if m.group(3) else _today().year
            return date(yy, mm, dd).isoformat()
        except ValueError:
            return None

    for w, idx in WEEKDAY_MAP.items():
        if re.search(rf"\b{re.escape(w)}\b", s, re.I):
            today = _today()
            delta = (idx - today.weekday()) % 7
            target = today if delta == 0 else today + timedelta(days=delta)
            return target.isoformat() if (target - today).days <= horizon_days else None
    return None


def parse_time_minutes(text: str) -> Optional[int]:
    """Parse a time of day into minutes-since-midnight. Returns None if none found."""
    raw = (text or "").strip()
    if not raw:
        return None
    s = normalize_text(raw)
    if "morning" in s:
        return 9 * 60
    if "afternoon" in s:
        return 13 * 60
    if "noon" in s:
        return 12 * 60

    m = _TIME_RE.search(raw)
    if not m:
        return None
    h = int(m.group(1))
    mins = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower() or None
    if not (0 <= mins <= 59) or not (1 <= h <= 24):
        return None
    if ampm:
        if h == 12 and ampm == "am":
            h = 0
        elif h != 12 and ampm == "pm":
            h += 12
    elif 1 <= h <= 7:  # bare "3" during business hours most likely means PM
        h += 12
    return h * 60 + mins if 0 <= h <= 23 else None


def parse_slot_choice(text: str, max_n: int) -> Optional[int]:
    """Parse a 1-based slot selection ('2', 'the third one') within 1..max_n."""
    t = (text or "").strip().lower()
    if not t:
        return None
    m = _SLOT_PICK_RE.search(t)
    if not m:
        return None
    for g in m.groups():
        if g and g.isdigit():
            n = int(g)
            return n if 1 <= n <= max_n else None
    for word, n in ORDINAL_MAP.items():
        if re.search(rf"\b{word}\b", t, re.I):
            return n if 1 <= n <= max_n else None
    return None


# =============================================================================
# SECTION 2 — Backends (swappable per office)
# Each backend is a small class with a clear method the engine calls. The
# "Null" versions are the safe default: they do nothing and let the engine fall
# through. Swap in a real implementation per office without touching the engine.
# =============================================================================

class SessionStore:
    """Keeps conversation state, keyed by (tenant_id, session_id).

    Default is in-memory. For production swap in a Redis/DB-backed subclass so
    state survives restarts and is shared across worker processes.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(self, tenant_id: str, session_id: str) -> Dict[str, Any]:
        return self._data.setdefault((tenant_id, session_id), {})

    def save(self, tenant_id: str, session_id: str, state: Dict[str, Any]) -> None:
        self._data[(tenant_id, session_id)] = state

    def clear(self, tenant_id: str, session_id: str) -> None:
        self._data.pop((tenant_id, session_id), None)


class IntentClassifier:
    """Optional ML intent service. Base version always says 'unknown'."""

    def classify(self, text: str) -> Dict[str, Any]:
        return {"intent": "other", "confidence": 0.0, "accepted": False}


class HttpIntentClassifier(IntentClassifier):
    def __init__(self, url: str, min_conf: float = 0.6, timeout: float = 5.0) -> None:
        self.url = url
        self.min_conf = min_conf
        self.timeout = timeout

    def classify(self, text: str) -> Dict[str, Any]:
        if not self.url:
            return super().classify(text)
        try:
            r = requests.post(self.url, json={"text": text}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json() or {}
            intent = data.get("coarse_intent") or data.get("intent") or "other"
            conf = float(data.get("confidence") or 0.0)
            accepted = bool(conf >= self.min_conf and intent and intent != "other")
            return {
                "intent": intent if accepted else "other",
                "confidence": conf,
                "accepted": accepted,
            }
        except Exception:
            return super().classify(text)


class RagProvider:
    """Optional knowledge-base retrieval. Base version returns nothing."""

    def answer(self, text: str, intent: str) -> Optional[str]:
        return None

    def retrieve(self, text: str, k: int = 4) -> List[Tuple[str, str, float]]:
        """Top-k (question, answer, score) — the CONTEXT handed to the LLM.

        This is the difference between retrieve-and-return (hand back a canned
        answer or nothing) and retrieve-and-GENERATE (give the model the facts
        and let it compose an answer). The second handles phrasings your KB
        never anticipated, without inventing anything.
        """
        return []


class HttpRagProvider(RagProvider):
    def __init__(self, url: str, timeout: float = 15.0) -> None:
        self.url = url
        self.timeout = timeout

    def answer(self, text: str, intent: str) -> Optional[str]:
        if not self.url:
            return None
        try:
            r = requests.post(
                self.url, json={"text": text, "intent": intent}, timeout=self.timeout
            )
            r.raise_for_status()
            ans = (r.json() or {}).get("answer", "").strip()
            return ans or None
        except Exception:
            return None


def parse_qa_file(text: str) -> List[Tuple[str, str]]:
    """Parse one knowledge file into (question, answer) pairs.

    Handles two formats automatically:
      * JSONL  — one JSON object per line with q/question and a/answer keys
      * Plain  — 'Q: ...' / 'A: ...' blocks (answers may span multiple lines;
                 blocks optionally separated by '---')
    """
    pairs: List[Tuple[str, str]] = []
    stripped = text.lstrip()
    if stripped.startswith("{"):  # JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (o.get("q") or o.get("question") or "").strip()
            a = (o.get("a") or o.get("answer") or "").strip()
            if q and a:
                pairs.append((q, a))
        return pairs
    # Plain Q/A: capture A until the next Q:, a '---' separator, or end of file
    for m in re.finditer(r"Q:\s*(.*?)\s*\nA:\s*(.*?)(?=\n\s*(?:Q:|---)|\Z)", text, re.S):
        q = " ".join(m.group(1).split())
        a = " ".join(m.group(2).split())
        if q and a:
            pairs.append((q, a))
    return pairs


class SemanticRagProvider(RagProvider):
    """Retrieval that matches MEANING, not keywords.

    The TF-IDF retriever below counts shared words, so it cannot connect
    "is someone required to pick me up" to "Do I need a driver?" — same meaning,
    almost no shared words. This encodes questions as sentence embeddings, so
    paraphrases land near each other in vector space and actually match.

    Needs:  pip install sentence-transformers
    Model:  all-MiniLM-L6-v2 (~80MB, fast on CPU, much faster on your GPU)

    If sentence-transformers isn't installed it returns None and the engine
    falls through — so this can never break a running system.
    """

    def __init__(self, knowledge_dir: str, min_similarity: float = 0.45,
                 model_name: str = "all-MiniLM-L6-v2") -> None:
        self.knowledge_dir = knowledge_dir
        self.min_similarity = min_similarity   # cosine on embeddings; ~0.45 is a sane start
        self.model_name = model_name
        self._ready = False
        self._model = None
        self._emb = None
        self._answers: List[str] = []
        self._questions: List[str] = []

    def _build_index(self) -> None:
        self._ready = True
        pairs: List[Tuple[str, str]] = []
        if os.path.isdir(self.knowledge_dir):
            for fn in sorted(os.listdir(self.knowledge_dir)):
                if not fn.lower().endswith(".txt"):
                    continue
                try:
                    with open(os.path.join(self.knowledge_dir, fn), encoding="utf-8", errors="ignore") as f:
                        pairs.extend(parse_qa_file(f.read()))
                except OSError:
                    continue
        if not pairs:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[rag] sentence-transformers not installed; semantic RAG is OFF.")
            print("[rag]   pip install sentence-transformers")
            return

        self._questions = [q for q, _ in pairs]
        self._answers = [a for _, a in pairs]
        self._model = SentenceTransformer(self.model_name)
        # Embed the KB's QUESTIONS: callers ask questions, so question-to-question
        # is the comparison that matters.
        self._emb = self._model.encode(self._questions, normalize_embeddings=True,
                                       show_progress_bar=False)
        print(f"[rag] semantic index ready: {len(pairs)} entries ({self.model_name})")

    def answer(self, text: str, intent: str) -> Optional[str]:
        if not self._ready:
            self._build_index()
        if self._model is None or not self._answers:
            return None
        import numpy as np
        q = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = (self._emb @ q[0])          # cosine, since both are normalized
        best = int(np.argmax(sims))
        if float(sims[best]) < self.min_similarity:
            return None
        return self._answers[best]

    def retrieve(self, text: str, k: int = 4) -> List[Tuple[str, str, float]]:
        if not self._ready:
            self._build_index()
        if self._model is None or not self._answers:
            return []
        import numpy as np
        q = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sims = (self._emb @ q[0])
        idx = np.argsort(sims)[::-1][:k]
        # A low bar here on purpose: this is CONTEXT for the model to read, not
        # an answer to speak. The model decides what's relevant; if none of it
        # is, its instructions tell it to say it doesn't know.
        return [(self._questions[i], self._answers[i], float(sims[i]))
                for i in idx if float(sims[i]) > 0.20]


class LocalRagProvider(RagProvider):
    """In-process TF-IDF retrieval over a per-office knowledge folder.

    Reads every *.txt file in `knowledge_dir`, indexes the Q/A pairs, and on
    each query returns the single best-matching answer if its similarity clears
    `min_similarity` (otherwise None, so the engine falls through). The index is
    built lazily on first use and cached. scikit-learn is imported lazily so the
    engine runs without it unless local RAG is actually enabled.

    If `kb_map` is provided and the ML classifier supplies an intent, retrieval
    is tried FIRST against only that intent's knowledge files. This is what stops
    a prep question from being answered out of the insurance file. If the in-lane
    search finds nothing, it falls back to searching the whole KB, so a wrong
    intent guess costs precision but never an answer.
    """

    def __init__(self, knowledge_dir: str, min_similarity: float = 0.18,
                 kb_map: Optional[Dict[str, List[str]]] = None) -> None:
        self.knowledge_dir = knowledge_dir
        self.min_similarity = min_similarity
        self.kb_map = kb_map or {}
        self._ready = False
        self._vectorizer = None
        self._matrix = None
        self._answers: List[str] = []
        self._questions: List[str] = []
        self._sources: List[str] = []  # which file each entry came from

    def _build_index(self) -> None:
        self._ready = True
        pairs: List[Tuple[str, str]] = []
        sources: List[str] = []
        if os.path.isdir(self.knowledge_dir):
            for fn in sorted(os.listdir(self.knowledge_dir)):
                if not fn.lower().endswith(".txt"):
                    continue
                try:
                    with open(os.path.join(self.knowledge_dir, fn), encoding="utf-8", errors="ignore") as f:
                        got = parse_qa_file(f.read())
                except OSError:
                    continue
                pairs.extend(got)
                sources.extend([fn] * len(got))
        if not pairs:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        except ImportError:
            return  # sklearn not installed -> local RAG stays off
        # Conversational filler must not drive retrieval. Without this, a KB
        # entry titled "What was the medication question?" scores highly against
        # ANY sentence containing the word "question" — which is most of them.
        stop = list(ENGLISH_STOP_WORDS) + [
            "question", "questions", "asking", "ask", "calling", "call",
            "referring", "wondering", "specific", "want", "know", "tell",
            "help", "please", "thanks", "thank", "hello", "yes",
        ]
        # Callers ask questions, so match question-to-question. Weight the KB's
        # question text (x3) over its answer text: otherwise a short, dense
        # answer out-scores a longer, more helpful one purely on term density.
        corpus = [f"{q} {q} {q} {a}" for q, a in pairs]
        self._questions = [q for q, _ in pairs]
        self._answers = [a for _, a in pairs]
        self._sources = sources
        self._vectorizer = TfidfVectorizer(stop_words=stop)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _best(self, text: str, allowed: Optional[set] = None) -> Optional[str]:
        """Best answer above threshold, optionally restricted to certain files."""
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        best_i, best_s = -1, -1.0
        for i, s in enumerate(sims):
            if allowed is not None and self._sources[i] not in allowed:
                continue
            if s > best_s:
                best_i, best_s = i, float(s)
        if best_i < 0 or best_s < self.min_similarity:
            return None
        return self._answers[best_i]

    def answer(self, text: str, intent: str) -> Optional[str]:
        if not self._ready:
            self._build_index()
        if self._vectorizer is None or not self._answers:
            return None
        # 1) If the classifier gave us a usable intent, search that lane first.
        lane_files = self.kb_map.get(intent or "")
        if lane_files:
            hit = self._best(text, allowed=set(lane_files))
            if hit:
                return hit
        # 2) Otherwise (or if the lane had nothing), search the whole KB.
        return self._best(text)

    def retrieve(self, text: str, k: int = 4) -> List[Tuple[str, str, float]]:
        """Top-k context for the LLM. Works even without sentence-transformers,
        though semantic retrieval finds far better context."""
        if not self._ready:
            self._build_index()
        if self._vectorizer is None or not self._answers:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sims = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(self._questions[i], self._answers[i], float(sims[i]))
                for i in idx if float(sims[i]) > 0.08]


class LlmProvider:
    """Optional general LLM (OpenAI-compatible chat completions). Base = off."""

    def generate(self, system_prompt: str, text: str) -> Optional[str]:
        return None

    def generate_grounded(self, system_prompt: str, text: str,
                          context: List[Tuple[str, str, float]]) -> Optional[str]:
        """Answer USING retrieved knowledge, not from the model's memory.

        This is the difference between a model that invents "the office is open
        8-6" and one that reads the actual office hours and rephrases them. The
        model's job becomes comprehension (which it is good at) rather than
        recall (which it hallucinates).
        """
        return None


class OpenAICompatLlm(LlmProvider):
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 120.0) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, system_prompt: str, text: str) -> Optional[str]:
        return self._call(system_prompt, text)

    def generate_grounded(self, system_prompt: str, text: str,
                          context: List[Tuple[str, str, float]]) -> Optional[str]:
        if not context:
            return None
        facts = "\n".join(f"- Q: {q}\n  A: {a}" for q, a, _ in context)
        prompt = (
            "Here is what the office's records say:\n\n"
            f"{facts}\n\n"
            f"The caller asked: \"{text}\"\n\n"
            "Answer the caller using ONLY the records above. If the records "
            "don't cover what they asked, say you don't have that information "
            "and offer to have the office follow up — do NOT guess. "
            "Reply in one or two short, warm sentences, as the front desk."
        )
        return self._call(system_prompt, prompt)

    def _call(self, system_prompt: str, user_content: str) -> Optional[str]:
        if not self.url:
            return None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 160,
            "stream": False,
        }
        try:
            r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return content or None
        except Exception:
            return None


class CalendarProvider:
    """Optional scheduling backend. Base version has no availability, so the
    engine gracefully falls back to the email path. Implement these two methods
    against a real calendar (Google, a clinic EHR, a reservations API, etc.)."""

    available = False

    def free_slots(self, day_iso: str) -> List[Dict[str, Any]]:
        """Return free slots for a given ISO date. Each slot: {'start': ISO}."""
        return []

    def book(self, start_iso: str, patient: Dict[str, Any]) -> bool:
        return False


class LocalCalendar(CalendarProvider):
    """File-backed calendar for a single office.

    Auto-seeds bookable slots for the allowed weekdays/hours into a per-tenant
    JSON file, lists free slots per day, and books them atomically. It's a real,
    working scheduler with no external dependencies — swap it for a Google
    Calendar or EHR-backed provider later by subclassing CalendarProvider the
    same way.
    """

    available = True
    _WD = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

    def __init__(self, path: str, day_start_hour: int = 9, day_end_hour: int = 17,
                 slot_minutes: int = 30, days: Optional[List[str]] = None,
                 seed_days: int = 28) -> None:
        self.path = path
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour
        self.slot_minutes = slot_minutes
        self.days = [d.lower()[:3] for d in (days or ["mon", "tue", "wed", "thu", "fri"])]
        self.seed_days = seed_days
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)  # atomic

    def _seed(self, store: Dict[str, Any]) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        allowed = {self._WD[d] for d in self.days if d in self._WD}
        today = datetime.now().date()
        for d in range(self.seed_days):
            day = today + timedelta(days=d)
            if day.weekday() not in allowed:
                continue
            minute = self.day_start_hour * 60
            while minute < self.day_end_hour * 60:
                h, m = divmod(minute, 60)
                start = datetime(day.year, day.month, day.day, h, m).isoformat(timespec="minutes")
                slots[start] = {"booked": False}
                minute += self.slot_minutes
        store["slots"] = slots
        return store

    def free_slots(self, day_iso: str) -> List[Dict[str, Any]]:
        with self._lock:
            store = self._load()
            if not store.get("slots"):
                store = self._seed(store)
                self._save(store)
            out = [{"start": s} for s, meta in store["slots"].items()
                   if not meta.get("booked") and s[:10] == day_iso]
            out.sort(key=lambda s: s["start"])
            return out

    def book(self, start_iso: str, patient: Dict[str, Any]) -> bool:
        with self._lock:
            store = self._load()
            if not store.get("slots"):
                store = self._seed(store)
            slots = store["slots"]
            key = start_iso if start_iso in slots else start_iso[:16]
            meta = slots.get(key)
            if not meta or meta.get("booked"):
                return False
            meta.update(booked=True, patient=patient,
                        booked_at=datetime.now().isoformat(timespec="seconds"))
            self._save(store)
            return True
# The profile is the "external needed info" for one office. It's pure config:
# rules directory, templates, prompts, and which backends to use. The engine
# reads these fields; it never hard-codes them.
# =============================================================================

DEFAULT_TEMPLATES: Dict[str, str] = {
    "GREETING": "Hi! Thanks for calling. How can I help you today?",
    "CLARIFY": "Sure — what can I help you with today?",
    "CAL_ASK_DAY": "Which day works best for you?",
    "CAL_DAY_REASK": "What day works best? (For example: Monday or Jan 19)",
    "CAL_ASK_TIME": "Great — we have availability from 9:00 AM to 5:00 PM. What time works best?",
    "CAL_TIME_REASK": "What time between 9:00 AM and 5:00 PM works best? (For example: 2:15 PM)",
    "CAL_NO_SLOTS": "Sorry — I don't see openings that day. Which other day works for you?",
    "CAL_OFFER": "Here are the available times:\n{slots}\nReply with the number you'd like (for example: 1 or 2).",
    "CAL_PICK_NUMBER": "Please reply with the number of the slot you want (for example: 1, 2, or 3).",
    "CAL_ASK_NAME": "Great — I can hold that time. What is the patient's full name?",
    "CAL_HOLD_AUTO": "Perfect — I can hold {time}. What is the patient's full name?",
    "CAL_ASK_DOB": "Thank you. What is the date of birth? You can say it as numbers — for example, four twenty-two nineteen seventy.",
    "CAL_DOB_REASK": "Sorry, I didn't catch that. Please say the date of birth slowly as numbers — month, day, year. For example: four, twenty-two, nineteen seventy.",
    "CAL_BOOKED": "You're all set — your appointment is booked.",
    "CAL_SLOT_GONE": "Sorry — that slot is no longer available. Which day works best for you?",
    "CAL_CANCELLED": "No problem — I've cancelled that booking request. What else can I help you with?",
    "CAL_GIVE_UP": "Sorry, I'm having trouble with that. Let's set the appointment aside for now — what else can I help you with?",
}


@dataclass
class Profile:
    """Everything specific to one office. Loaded from profiles/<tenant>/profile.json."""

    tenant_id: str
    display_name: str = "the office"
    system_prompt: str = "You are a polite front-desk assistant. Be brief and factual. Do not invent hours, addresses, insurance details, or clinical advice. If you don't know, say so."
    email_fallback: str = "Please email your question to the office and we'll get back to you."

    # where this office's .jsonl rules live, and the intent->files map
    rules_dir: str = ""
    intent_map: Dict[str, List[str]] = field(default_factory=dict)

    # office-specific ASR fixes (e.g. provider name mis-hears)
    asr_aliases: List[Tuple[str, str]] = field(default_factory=list)

    # response strings (falls back to DEFAULT_TEMPLATES)
    templates: Dict[str, str] = field(default_factory=dict)

    # scheduling window
    day_start_hour: int = 9
    day_end_hour: int = 17
    horizon_days: int = 14

    # backends (constructed by the store from the raw config)
    intent: IntentClassifier = field(default_factory=IntentClassifier)
    rag: RagProvider = field(default_factory=RagProvider)
    llm: LlmProvider = field(default_factory=LlmProvider)
    calendar: CalendarProvider = field(default_factory=CalendarProvider)

    def t(self, key: str, **fmt: Any) -> str:
        """Look up a response template, with default fallback and formatting."""
        raw = self.templates.get(key) or DEFAULT_TEMPLATES.get(key, "")
        try:
            return raw.format(**fmt) if fmt else raw
        except Exception:
            return raw


class ProfileStore:
    """Loads a Profile per tenant from disk and caches it, reloading when the
    profile file changes on disk (mtime check) so edits take effect without a
    restart."""

    def __init__(self, profiles_dir: str) -> None:
        self.profiles_dir = profiles_dir
        self._cache: Dict[str, Tuple[float, Profile]] = {}

    def _path(self, tenant_id: str) -> str:
        return os.path.join(self.profiles_dir, tenant_id, "profile.json")

    def get(self, tenant_id: str) -> Profile:
        path = self._path(tenant_id)
        mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
        cached = self._cache.get(tenant_id)
        if cached and cached[0] == mtime:
            return cached[1]
        profile = self._build(tenant_id, path)
        self._cache[tenant_id] = (mtime, profile)
        return profile

    def _build(self, tenant_id: str, path: str) -> Profile:
        raw: Dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
            except Exception:
                raw = {}

        rules_dir = raw.get("rules_dir") or os.path.join(self.profiles_dir, tenant_id, "rules")

        # Build backends from the "backends" section of the config.
        b = raw.get("backends", {})
        intent_cfg = b.get("intent", {})
        rag_cfg = b.get("rag", {})
        llm_cfg = b.get("llm", {})

        intent = (
            HttpIntentClassifier(
                intent_cfg["url"],
                min_conf=float(intent_cfg.get("min_conf", 0.6)),
            )
            if intent_cfg.get("url")
            else IntentClassifier()
        )
        rag_type = rag_cfg.get("type")
        if rag_type in ("semantic", "local"):
            knowledge_dir = rag_cfg.get("knowledge_dir") or os.path.join(self.profiles_dir, tenant_id, "knowledge")
            if rag_type == "semantic":
                # Understands meaning: "is someone required to pick me up" ==
                # "do I need a driver". Needs sentence-transformers installed.
                rag = SemanticRagProvider(
                    knowledge_dir,
                    min_similarity=float(rag_cfg.get("min_similarity", 0.45)),
                    model_name=rag_cfg.get("model", "all-MiniLM-L6-v2"),
                )
            else:
                rag = LocalRagProvider(
                    knowledge_dir,
                    min_similarity=float(rag_cfg.get("min_similarity", 0.35)),
                    kb_map=raw.get("kb_map") or {},
                )
        elif rag_cfg.get("url"):
            rag = HttpRagProvider(rag_cfg["url"])
        else:
            rag = RagProvider()
        llm = (
            OpenAICompatLlm(
                llm_cfg["url"], llm_cfg.get("model", "local-model"), llm_cfg.get("api_key", "")
            )
            if llm_cfg.get("url")
            else LlmProvider()
        )
        # Calendar: file-backed local scheduler when configured, else off.
        cal_cfg = raw.get("calendar", {})
        if cal_cfg.get("type") == "local":
            cal_path = cal_cfg.get("path") or os.path.join(self.profiles_dir, tenant_id, "calendar.json")
            calendar = LocalCalendar(
                cal_path,
                day_start_hour=int(raw.get("day_start_hour", 9)),
                day_end_hour=int(raw.get("day_end_hour", 17)),
                slot_minutes=int(cal_cfg.get("slot_minutes", 30)),
                days=cal_cfg.get("days"),
                seed_days=int(cal_cfg.get("seed_days", 28)),
            )
        else:
            calendar = CalendarProvider()

        return Profile(
            tenant_id=tenant_id,
            display_name=raw.get("display_name", "the office"),
            system_prompt=raw.get("system_prompt", Profile.__dataclass_fields__["system_prompt"].default),
            email_fallback=raw.get("email_fallback", Profile.__dataclass_fields__["email_fallback"].default),
            rules_dir=rules_dir,
            intent_map={k: (v if isinstance(v, list) else [v]) for k, v in (raw.get("intent_map") or {}).items()},
            asr_aliases=[tuple(x) for x in raw.get("asr_aliases", [])],
            templates=raw.get("templates", {}),
            day_start_hour=int(raw.get("day_start_hour", 9)),
            day_end_hour=int(raw.get("day_end_hour", 17)),
            horizon_days=int(raw.get("horizon_days", 14)),
            intent=intent,
            rag=rag,
            llm=llm,
            calendar=calendar,
        )


# =============================================================================
# SECTION 4 — FSM rules (JSONL loading + matching)
# =============================================================================

@dataclass
class Rule:
    id: str
    intent: str
    patterns: List[str]
    response: str
    origin: str


_RULE_CACHE: Dict[str, Tuple[float, List[Rule]]] = {}


def load_rules(path: str) -> List[Rule]:
    """Load rules from a .jsonl file, cached with mtime invalidation."""
    if not os.path.isfile(path):
        return []
    mtime = os.path.getmtime(path)
    cached = _RULE_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    rules: List[Rule] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id") or ""
                patterns = obj.get("patterns") or []
                resp = obj.get("response") or ""
                if rid and isinstance(patterns, list) and patterns and resp:
                    rules.append(Rule(rid, obj.get("intent", ""), patterns, resp, origin=path))
    except Exception:
        rules = []
    _RULE_CACHE[path] = (mtime, rules)
    return rules


def match_rules(norm_text: str, rules: List[Rule]) -> Optional[Rule]:
    if not norm_text:
        return None
    for r in rules:
        for pat in r.patterns:
            try:
                if re.search(pat, norm_text, re.I):
                    return r
            except re.error:
                continue
    return None


def _is_special(filename: str) -> bool:
    # greeting.jsonl is handled by the dedicated greet-once path; clarify and
    # fallback are handled by profile templates. All three are kept out of the
    # general rule sweep so they don't fire on every matching message.
    fn = filename.lower()
    return "clarify" in fn or "fallback" in fn or "greeting" in fn


def list_rule_files(rules_dir: str) -> List[str]:
    """Rule files in sweep order.

    SAFETY: triage/urgent files are swept FIRST. Symptom reports must never be
    caught by a topical rule that happens to share a keyword — e.g. "I have a
    fever after my colonoscopy" must reach triage, not a prep-rescheduling rule
    that also matches "fever".
    """
    out: List[str] = []
    try:
        for fn in sorted(os.listdir(rules_dir)):
            if fn.lower().endswith(".jsonl") and not _is_special(fn):
                fp = os.path.join(rules_dir, fn)
                if os.path.isfile(fp):
                    out.append(fp)
    except FileNotFoundError:
        pass

    def priority(path: str) -> int:
        name = os.path.basename(path).lower()
        return 0 if ("triage" in name or "urgent" in name) else 1

    out.sort(key=lambda p: (priority(p), p))
    return out


# =============================================================================
# SECTION 5 — The Engine (the cascade)
# =============================================================================

class Engine:
    def __init__(self, profiles: ProfileStore, sessions: SessionStore) -> None:
        self.profiles = profiles
        self.sessions = sessions

    def handle(self, tenant_id: str, session_id: str, text: str) -> Dict[str, Any]:
        profile = self.profiles.get(tenant_id)
        state = self.sessions.get(tenant_id, session_id)
        text = (text or "").strip()
        norm = normalize_text(text, extra_aliases=profile.asr_aliases)

        # ---- 1. Mid-conversation scheduling state machine -------------------
        if state.get("awaiting"):
            # Escape hatch 1: caller explicitly wants out. But weak words like
            # "wait"/"hold on" shouldn't cancel if the message is actually a
            # question we can answer ("wait, do I need a driver?").
            if CANCEL_RE.search(text) and not self._answerable(profile, norm):
                state.clear()
                self.sessions.save(tenant_id, session_id, state)
                return self._out(profile.t("CAL_CANCELLED"), source="calendar", lane="scheduling")

            reply = self._continue_scheduling(profile, state, text, norm)
            self.sessions.save(tenant_id, session_id, state)
            if reply is not None:
                return reply
            # reply is None -> the caller asked something else, or we gave up.
            # Fall through and let the normal cascade answer them.

        # ---- 2. FSM rule match (greeting first, then all rule files) --------
        greet = match_rules(norm, load_rules(os.path.join(profile.rules_dir, "greeting.jsonl")))
        if greet and not state.get("greeted"):
            state["greeted"] = True
            self.sessions.save(tenant_id, session_id, state)
            return self._out(greet.response, source="fsm", lane="greeting", rule_id=greet.id)

        for fp in list_rule_files(profile.rules_dir):
            hit = match_rules(norm, load_rules(fp))
            if hit:
                if hit.intent in {"appointment_scheduling", "appointment_info"} or SCHED_INTENT_RE.search(norm):
                    return self._start_scheduling(profile, state, tenant_id, session_id)
                return self._out(hit.response, source="fsm", lane="rules", rule_id=hit.id)

        # ---- 3. Scheduling regex (rules missed it) --------------------------
        if SCHED_INTENT_RE.search(norm):
            return self._start_scheduling(profile, state, tenant_id, session_id)

        # ---- 4. Fragment / clarify gate -------------------------------------
        if is_fragment(text) and not is_smalltalk(text):
            return self._out(profile.t("CLARIFY"), source="fsm", lane="clarify")

        # ---- 5. ML intent classify ------------------------------------------
        ml = profile.intent.classify(text)
        lane = ml["intent"] if ml["accepted"] else "other"

        # ---- 6. Lane-scoped rules (if intent accepted) ----------------------
        if ml["accepted"]:
            for fn in profile.intent_map.get(lane, []):
                hit = match_rules(norm, load_rules(os.path.join(profile.rules_dir, fn)))
                if hit:
                    return self._out(hit.response, source="fsm", lane=lane, rule_id=hit.id, ml=ml)

        # ---- 7. RAG: a confident, exact match answers directly ---------------
        rag_ans = profile.rag.answer(text, lane)
        if rag_ans:
            return self._out(rag_ans, source="rag", lane=lane, ml=ml)

        # ---- 7b. GROUNDED GENERATION -----------------------------------------
        # No single KB entry was a confident match — but the KB may still hold
        # the facts, phrased differently. ("is someone required to pick me up"
        # vs "Do I need a driver?") So retrieve the closest entries and let the
        # LLM compose an answer FROM THEM. It reads rather than recalls, which
        # is what stops it inventing office hours and insurance policies.
        #
        # CRITICAL: only ground on context that is actually relevant. Handing the
        # model unrelated entries produces a CONFIDENTLY WRONG answer built from
        # the wrong facts — worse than admitting ignorance. Weak retrieval (which
        # is what TF-IDF gives you on a paraphrase) must not be grounded on.
        # This is why `"rag": {"type": "semantic"}` is a prerequisite, not a
        # nice-to-have.
        context = profile.rag.retrieve(text, k=4)
        strong = [c for c in context if c[2] >= GROUNDING_MIN_SCORE]
        if strong:
            grounded = profile.llm.generate_grounded(profile.system_prompt, text, strong)
            if grounded and _llm_output_ok(grounded):
                return self._out(grounded, source="llm_rag", lane=lane, ml=ml)

        # ---- 8. LLM with NO grounding ---------------------------------------
        # OFF by default, and this is a hard-won default.
        #
        # A working instruction-following model, asked "Do you validate parking?"
        # with no facts to work from, replies:
        #
        #   "Yes, we do validate parking for our patients. Please bring your
        #    parking ticket to the front desk when you check in."
        #
        # Fluent. Warm. Correct-sounding. Entirely fabricated. The office does
        # not validate parking. The system prompt said "never invent facts" and
        # the model invented one anyway — because that is what language models
        # do when asked a question they have no information about.
        #
        # This is MORE dangerous than the old base model's obvious nonsense: a
        # caller can tell that a game-mod description is broken, but they cannot
        # tell that a confident parking policy is false. They just show up with
        # a ticket.
        #
        # So the model only ever speaks FROM RETRIEVED FACTS (step 7b). With no
        # facts, the honest answer is the email fallback.
        #
        # Set ALLOW_UNGROUNDED_LLM=1 only if you accept it will invent things.
        if ALLOW_UNGROUNDED_LLM:
            llm_ans = profile.llm.generate(profile.system_prompt, text)
            if llm_ans and _llm_output_ok(llm_ans):
                return self._out(llm_ans, source="llm", lane=lane, ml=ml)

        # ---- 9. Final fallback ----------------------------------------------
        return self._out(profile.email_fallback, source="fallback", lane=lane, matched=False, ml=ml)

    # ---- scheduling helpers -------------------------------------------------

    def _start_scheduling(self, profile: Profile, state: Dict[str, Any],
                          tenant_id: str, session_id: str) -> Dict[str, Any]:
        if not profile.calendar.available:
            return self._out(profile.email_fallback, source="fallback", lane="scheduling", matched=False)
        state["awaiting"] = "day"
        self.sessions.save(tenant_id, session_id, state)
        return self._out(profile.t("CAL_ASK_DAY"), source="calendar", lane="scheduling")

    @staticmethod
    def _answerable(profile: Profile, norm: str) -> bool:
        """True if this message matches a rule or the knowledge base — i.e. it's
        a real question we could answer, not just noise or a cancel."""
        for fp in list_rule_files(profile.rules_dir):
            if match_rules(norm, load_rules(fp)):
                return True
        return bool(profile.rag.answer(norm, "other"))

    def _reask(self, profile: Profile, state: Dict[str, Any], norm: str,
               template_key: str) -> Optional[Dict[str, Any]]:
        """A scheduling step failed to parse. Decide: re-ask, bail to the
        cascade (they asked something else), or give up gracefully."""
        # Escape hatch 2: does this look like a different question we can answer?
        if self._answerable(profile, norm):
            state.clear()
            return None  # -> main cascade answers it

        # Escape hatch 3: don't badger. After a couple of tries, let them go.
        state["reasks"] = state.get("reasks", 0) + 1
        if state["reasks"] > MAX_REASKS:
            state.clear()
            return self._out(profile.t("CAL_GIVE_UP"), source="calendar", lane="scheduling")
        return self._out(profile.t(template_key), source="calendar", lane="scheduling")

    def _continue_scheduling(self, profile: Profile, state: Dict[str, Any],
                             text: str, norm: str = "") -> Optional[Dict[str, Any]]:
        step = state.get("awaiting")

        if step == "day":
            day = parse_day_choice(text, horizon_days=max(profile.horizon_days, 30))
            if not day:
                return self._reask(profile, state, norm, "CAL_DAY_REASK")
            state["day"] = day
            state["awaiting"] = "time"
            state["reasks"] = 0
            return self._out(profile.t("CAL_ASK_TIME"), source="calendar", lane="scheduling")

        if step == "time":
            mins = parse_time_minutes(text)
            window = profile.day_start_hour * 60 <= (mins or -1) <= profile.day_end_hour * 60
            if mins is None or not window:
                return self._reask(profile, state, norm, "CAL_TIME_REASK")
            state["reasks"] = 0
            return self._offer_or_hold(profile, state, mins)

        if step == "slot_pick":
            offered = state.get("offered", [])
            choice = parse_slot_choice(text, max_n=len(offered))
            if choice is None:
                return self._reask(profile, state, norm, "CAL_PICK_NUMBER")
            state["pending_start"] = offered[choice - 1]["start"]
            state["awaiting"] = "name"
            state["reasks"] = 0
            return self._out(profile.t("CAL_ASK_NAME"), source="calendar", lane="scheduling")

        if step == "name":
            name = text.strip()
            # A name can be almost anything, so the only reliable signal that
            # the caller has changed the subject is that they asked a QUESTION
            # ("wait, do I need a driver?"). Do NOT also test whether the text
            # matches a rule: real names contain ordinary words — "Test Patient",
            # "Bill Banks", "Mark Payne" — and would be thrown out of the booking.
            if _QUESTIONY_RE.search(name):
                return self._reask(profile, state, norm, "CAL_ASK_NAME")
            if len(name) < 2:
                return self._reask(profile, state, norm, "CAL_ASK_NAME")
            state["name"] = name
            state["awaiting"] = "dob"
            state["reasks"] = 0
            # pii=True: the caller's input on this turn WAS the patient name.
            return self._out(profile.t("CAL_ASK_DOB"), source="calendar",
                             lane="scheduling", pii=True)

        if step == "dob":
            dob = parse_dob_iso(text)
            if not dob:
                return self._reask(profile, state, norm, "CAL_DOB_REASK")
            ok = False
            start = state.get("pending_start")
            if start:
                ok = profile.calendar.book(start, {"name": state.get("name"), "dob": dob})
            state.clear()
            if ok:
                # pii=True: the caller's input on this turn WAS a date of birth.
                return self._out(profile.t("CAL_BOOKED"), source="calendar",
                                 lane="scheduling", pii=True)
            state["awaiting"] = "day"
            return self._out(profile.t("CAL_SLOT_GONE"), source="calendar",
                             lane="scheduling", pii=True)

        return None  # unknown step -> let the main cascade handle it

    def _offer_or_hold(self, profile: Profile, state: Dict[str, Any], pref_mins: int) -> Dict[str, Any]:
        free = profile.calendar.free_slots(state["day"])
        if not free:
            state["awaiting"] = "day"
            return self._out(profile.t("CAL_NO_SLOTS"), source="calendar", lane="scheduling")

        def slot_mins(slot: Dict[str, Any]) -> int:
            try:
                dt = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
                return dt.hour * 60 + dt.minute
            except Exception:
                return 10 ** 9

        free.sort(key=lambda s: abs(slot_mins(s) - pref_mins))
        best = free[0]
        if abs(slot_mins(best) - pref_mins) <= 30:
            state["pending_start"] = best["start"]
            state["awaiting"] = "name"
            return self._out(profile.t("CAL_HOLD_AUTO", time=_fmt_slot(best["start"])),
                             source="calendar", lane="scheduling")

        options = free[:8]
        state["offered"] = options
        state["awaiting"] = "slot_pick"
        slots_txt = "\n".join(f"{i}) {_fmt_slot(s['start'])}" for i, s in enumerate(options, 1))
        return self._out(profile.t("CAL_OFFER", slots=slots_txt), source="calendar", lane="scheduling")

    # ---- response builder ---------------------------------------------------

    @staticmethod
    def _out(text: str, *, source: str, lane: str, matched: bool = True,
             rule_id: Optional[str] = None, ml: Optional[Dict[str, Any]] = None,
             pii: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": True,
            "response": text,
            "source": source,
            "lane": lane,
            "matched": matched,
            "rule_id": rule_id,
            "used_ml": bool(ml and ml.get("accepted")),
            # True when the caller's input on THIS turn was patient identity
            # data (name / DOB). Logging uses this to redact by context.
            "pii": pii,
        }
        if ml:
            out["ml_intent"] = ml.get("intent")
            out["ml_confidence"] = ml.get("confidence")
        return out


# =============================================================================
# SECTION 5b — LLM output guard
# =============================================================================

# A misconfigured or base (non-instruct) model doesn't refuse — it CONTINUES
# text. It will happily produce forum posts, game-mod descriptions, or invented
# medical advice. The system prompt is a request, not a guarantee, so the engine
# checks the output before ever speaking it to a caller.

_LLM_JUNK_RE = re.compile(
    r"\b(subreddit|reddit|upvot|downvot|forum|thread|blog post|"
    r"click here|subscribe|my name is \w+ and i|"
    r"this is a mod\b|minecraft|game|steam|"
    r"http[s]?://|www\.|"
    r"as an ai language model|i am an ai)\b",
    re.I,
)

# The model must speak AS THE OFFICE ("your doctor", "we can..."), never as a
# patient. A base model will happily role-play a caller — e.g. "I've been trying
# to get MY doctor to come to the hospital when I am in labor. MY midwife said..."
# That is first-person patient narrative and must never reach a caller.
_LLM_ROLEPLAY_RE = re.compile(
    r"\bmy (doctor|midwife|physician|gastroenterologist|appointment|procedure|"
    r"colonoscopy|endoscopy|insurance|symptoms|surgery|husband|wife|mom|mother)\b"
    r"|\bi'?(ve| have) been (trying|playing|having|waiting|dealing)\b"
    r"|\bwhen i (am|was) in labor\b"
    r"|\bi'?m not sure if this is the right (place|sub)\b"
    r"|\bi (had|have) (my|a) (colonoscopy|endoscopy|procedure|appointment)\b"
    # ...and the model editorializing. A receptionist states office policy; it
    # never shares feelings or opinions ("I'm not sure how I feel about this").
    r"|\bi'?m not sure how i feel\b"
    r"|\bin my opinion\b"
    r"|\bi (think|feel|believe|guess) (it'?s|that|this|they|you'?d)\b"
    r"|\bit'?s a good idea to have\b"
    r"|\bpersonally,? i\b",
    re.I,
)

# Things the bot must never say on a medical line without grounding.
_LLM_UNSAFE_RE = re.compile(
    r"\b(you (should|can) take \d|dose of|mg of|milligrams|"
    r"you (probably |likely )?have\b|it'?s probably (nothing|fine)|"
    r"don'?t worry|no need to (see|call) (a )?(doctor|911)|"
    r"you don'?t need (to see|medical))\b",
    re.I,
)

# Chat-template tokens leaking into the answer = the model is confused.
_LLM_TOKEN_RE = re.compile(r"\[/?INST\]|<\|.*?\|>|</?s>", re.I)


def _llm_output_ok(text: str) -> bool:
    """Reject LLM output that is junk, unsafe, role-played, or off-task.

    Returning False makes the engine fall through to the email fallback — a
    boring, safe answer is always better than a confident wrong one.
    """
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if len(t) > 600:               # rambling continuation, not a receptionist reply
        return False
    if _LLM_TOKEN_RE.search(t):    # [INST] etc. leaking through
        return False
    if _LLM_JUNK_RE.search(t):     # continuing web text, not answering
        return False
    if _LLM_ROLEPLAY_RE.search(t):  # speaking as a patient, not as the office
        return False
    if _LLM_UNSAFE_RE.search(t):   # dosing / reassurance / diagnosis — never
        return False
    return True


# =============================================================================
# SECTION 5c — Conversation logging (for debugging; PHI-scrubbed by default)
# =============================================================================

# Off unless you ask for it. LOG_TURNS=1 to enable.
LOG_TURNS = os.environ.get("LOG_TURNS", "0") == "1"
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "turns.jsonl"))
# Scrub patient identifiers before writing. Only set to 0 in a private dev
# environment with fake data — raw logs of real calls are PHI at rest.
LOG_SCRUB = os.environ.get("LOG_SCRUB", "1") == "1"

try:
    from phi_scrub import redact_phi as _redact
except ImportError:  # phi_scrub.py not present -> fail safe: don't log raw PHI
    _redact = None


def log_turn(tenant: str, session: str, text_in: str, out: Dict[str, Any]) -> None:
    """Append one conversation turn to a JSONL file, for debugging.

    Records what the bot HEARD and which layer answered — the two things you
    need to diagnose 'it didn't understand me'. Scrubs names/DOBs/phones first
    unless explicitly disabled.
    """
    if not LOG_TURNS:
        return
    heard = text_in or ""
    said = out.get("response", "")
    if LOG_SCRUB:
        if _redact is None:
            heard = said = "[NOT LOGGED: phi_scrub.py missing]"
        else:
            # Context beats guesswork: if the engine knows this turn's input was
            # a patient name or DOB, drop it entirely rather than hoping a regex
            # recognizes it (a bare name like "Maria Gonzalez" has no giveaway).
            heard = "[REDACTED_PATIENT_DETAIL]" if out.get("pii") else _redact(heard)
            said = _redact(said)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tenant": tenant,
        "session": session,
        "heard": heard,
        "said": said,
        "source": out.get("source"),
        "lane": out.get("lane"),
        "rule_id": out.get("rule_id"),
        "matched": out.get("matched"),
    }
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass  # never let logging break a call


# =============================================================================
# SECTION 6 — Flask app
# =============================================================================

PROFILES_DIR = os.environ.get("PROFILES_DIR", os.path.join(os.path.dirname(__file__), "profiles"))
DEFAULT_TENANT = os.environ.get("DEFAULT_TENANT", "default")

app = Flask(__name__)
CORS(app)

_engine = Engine(ProfileStore(PROFILES_DIR), SessionStore())


def _tenant_and_session(payload: Dict[str, Any]) -> Tuple[str, str]:
    tenant = (payload.get("tenant_id") or payload.get("tenant") or DEFAULT_TENANT).strip()
    session = (payload.get("session_id") or payload.get("session") or "default").strip()
    return tenant or DEFAULT_TENANT, session or "default"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "profiles_dir": PROFILES_DIR})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    tenant, session = _tenant_and_session(payload)
    text = (payload.get("text") or payload.get("message") or payload.get("user_text") or "").strip()
    out = _engine.handle(tenant, session, text)
    log_turn(tenant, session, text, out)
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"[+] engine listening on http://0.0.0.0:{port}")
    print(f"[+] profiles dir: {PROFILES_DIR}")
    app.run(host="0.0.0.0", port=port, debug=False)
