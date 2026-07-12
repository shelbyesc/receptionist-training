#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_scrub.py — stronger PHI redaction for transcripts and knowledge files.

Your original redact_phi() caught phones, emails, and SSNs only — which is why
names and dates of birth leaked into the knowledge base. This module adds dates
(incl. DOBs), long ID/MRN digit runs, and name disclosures, and can optionally
use spaCy NER for free-floating names.

Two ways to use it:

1) In your pipeline (1_download_transcribe.py) — replace the import:
       from phi_scrub import redact_phi
   ...and delete the old redact_phi definition. Everything else stays.

2) As a CLI to clean files you already have (e.g. transcripts/ or a KB folder):
       python phi_scrub.py transcripts/ --in-place
       python phi_scrub.py profiles/default/knowledge/ --out cleaned/
       python phi_scrub.py file.txt            # prints cleaned text to stdout

Names: regex catches disclosed names ("my name is X", "this is X", "name: X").
Free-floating names (like "Yes, Salim Shelby.") need NER — install spaCy for
full coverage:  pip install spacy && python -m spacy download en_core_web_sm
Then pass --ner. Without it, structured identifiers are still redacted.
"""

import re
import os
import sys
import argparse
from typing import Optional

# ---- structured identifiers -------------------------------------------------
PHONE_RE = re.compile(r'(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
SSN_RE   = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
# long digit runs (MRN, member IDs) — 6+ digits, not already a phone/date
ID_RE    = re.compile(r'\b\d{6,}\b')

_MONTHS = (r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
           r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)')
# dates / DOBs in several formats
DATE_RE = re.compile(
    r'\b(?:'
    r'\d{4}-\d{1,2}-\d{1,2}'                              # 1970-04-22
    r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'                     # 04/22/1970
    r'|' + _MONTHS + r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}'  # April 22, 1970
    r'|\d{1,2}(?:st|nd|rd|th)?\s+' + _MONTHS + r'\.?,?\s+\d{2,4}'  # 22nd April 1970
    r')\b',
    re.IGNORECASE,
)

# name disclosures: trigger phrase is case-insensitive, but the captured name
# stays case-sensitive so it grabs only Capitalized name tokens (not "and my").
NAME_PHRASE_RE = re.compile(
    r'(?i:\b(?:my name is|this is|i am|i\'m|name is|patient(?:\'s)? name(?: is)?|'
    r'speaking with|caller is|name:))'
    r'(\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})'
)


def redact_phi(text: str, use_ner: bool = False) -> str:
    """Redact PHI from text. Order matters: dates/ids before names."""
    if not text:
        return text
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = SSN_RE.sub("[REDACTED_ID]", text)
    text = DATE_RE.sub("[REDACTED_DATE]", text)
    text = ID_RE.sub("[REDACTED_ID]", text)
    # keep the disclosure phrase + spacing, redact only the trailing name
    text = NAME_PHRASE_RE.sub(lambda m: m.group(0)[: -len(m.group(2))] + "[REDACTED_NAME]", text)
    if use_ner:
        text = _redact_names_ner(text)
    return text


_NLP = None


def _redact_names_ner(text: str) -> str:
    """Redact PERSON entities via spaCy if available; no-op otherwise."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            print("[phi_scrub] spaCy/en_core_web_sm not available; skipping NER names.",
                  file=sys.stderr)
            _NLP = False
    if not _NLP:
        return text
    doc = _NLP(text)
    out, last = [], 0
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            out.append(text[last:ent.start_char])
            out.append("[REDACTED_NAME]")
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


# ---- CLI --------------------------------------------------------------------
def _iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith((".txt", ".jsonl")):
                        yield os.path.join(root, fn)
        elif os.path.isfile(p):
            yield p


def main():
    ap = argparse.ArgumentParser(description="Redact PHI from text/transcript files.")
    ap.add_argument("paths", nargs="+", help="Files or folders to scrub")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the files")
    ap.add_argument("--out", help="Write cleaned copies into this folder")
    ap.add_argument("--ner", action="store_true", help="Also redact names via spaCy NER")
    args = ap.parse_args()

    files = list(_iter_files(args.paths))
    if not files:
        print("No .txt/.jsonl files found.", file=sys.stderr)
        return

    if not args.in_place and not args.out and len(files) == 1:
        # single file, no destination -> print to stdout
        print(redact_phi(open(files[0], encoding="utf-8", errors="ignore").read(), use_ner=args.ner))
        return

    for fp in files:
        cleaned = redact_phi(open(fp, encoding="utf-8", errors="ignore").read(), use_ner=args.ner)
        if args.in_place:
            dst = fp
        elif args.out:
            dst = os.path.join(args.out, os.path.relpath(fp))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        else:
            dst = fp + ".redacted"
        with open(dst, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"scrubbed {fp} -> {dst}")


if __name__ == "__main__":
    main()
