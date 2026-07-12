#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Render-hosted TRAINING build of the receptionist.

*** FAKE DATA ONLY. NEVER type or speak a real patient's name, date of birth,
    or symptoms into this app. This is not a HIPAA-enabled deployment. ***

Why this exists: Render has no GPU, and Whisper + Kokoro + torch won't fit in a
small instance. They aren't needed here — the BROWSER does speech-to-text and
text-to-speech natively (Web Speech API). So Render only has to run the engine:
plain Flask + scikit-learn, no ML models, no GPU, fits the free tier.

    phone/laptop browser  --(browser STT)-->  text
                          --POST /chat----->  engine (rules + RAG + calendar)
                          <--reply text-----
                          --(browser TTS)-->  spoken aloud

Everything the caller says is recognised and spoken IN THEIR BROWSER. The only
thing crossing the network is text.

Local test:
    pip install -r requirements.txt
    python app.py            # http://localhost:10000

Deploy: push to a repo, connect it on Render as a Web Service (render.yaml
included), Build `pip install -r requirements.txt`, Start `python app.py`.
"""

import os
import json
import secrets
from datetime import datetime
from collections import Counter, deque

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from engine import Engine, ProfileStore, SessionStore
from phi_scrub import redact_phi

PORT = int(os.environ.get("PORT", "10000"))     # Render injects PORT
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
TENANT = os.environ.get("DEFAULT_TENANT", "default")

# Set ADMIN_TOKEN in Render's env vars to read /misses and /stats from a browser.
# Without it those endpoints are closed — this is a public URL.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = Flask(__name__)
CORS(app)

_engine = Engine(ProfileStore(PROFILES_DIR), SessionStore())

# In-memory only — Render's free tier has an ephemeral filesystem, so a log FILE
# would vanish on restart anyway. Turns go to stdout (captured privately in
# Render's dashboard) AND into a ring buffer you can pull down to your PC via
# /export. Pull them regularly: a restart clears the buffer.
_counts = Counter()
_misses = deque(maxlen=200)
_turns = deque(maxlen=5000)      # what /export hands back


def _log_turn(session: str, text: str, out: dict) -> None:
    """Log a turn to stdout and the export buffer, PHI-scrubbed twice over.

    Render's own HIPAA guidance says never put PHI in logs. This is a public
    training URL, so even though it's meant for fake data, we assume someone
    will eventually type something real:
      1. If the engine KNOWS this turn was a name or DOB (the booking flow just
         asked for one), the input is dropped entirely — no guessing.
      2. Everything else goes through the regex scrubber (phones, emails, dates,
         IDs, disclosed names).
    """
    heard = "[REDACTED_PATIENT_DETAIL]" if out.get("pii") else redact_phi(text)
    said = redact_phi(out.get("response", ""))[:120]
    source = out.get("source", "?")

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session": session,
        "heard": heard,
        "said": said,
        "source": source,
        "lane": out.get("lane"),
        "rule_id": out.get("rule_id"),
        "matched": out.get("matched"),
    }

    _counts[source] += 1
    _turns.append(row)
    if source == "fallback":
        _misses.append({"ts": row["ts"], "heard": heard})

    print(json.dumps(row, ensure_ascii=False), flush=True)


def _authed() -> bool:
    if not ADMIN_TOKEN:
        return False
    return request.args.get("token") == ADMIN_TOKEN


@app.route("/health")
def health():
    return jsonify({"ok": True, "mode": "training", "phi": "fake data only"})


@app.route("/stats")
def stats():
    """How often each layer answers. No conversation text — safe to expose."""
    total = sum(_counts.values())
    if not total:
        return jsonify({"turns": 0})
    return jsonify({
        "turns": total,
        "by_source": dict(_counts),
        "fallback_rate": round(_counts["fallback"] / total, 3),
    })


@app.route("/misses")
def misses():
    """Questions the bot could NOT answer — your to-do list.

    Token-protected: this echoes back what people typed, and the URL is public.
    """
    if not _authed():
        return jsonify({"error": "set ADMIN_TOKEN in Render, then call /misses?token=..."}), 403
    return jsonify({"count": len(_misses), "misses": list(_misses)})


@app.route("/export")
def export():
    """Download every logged turn as JSONL — the same format review_log.py reads.

    Token-protected: it echoes back what people typed, and this URL is public.
    Pull it with fetch_logs.py on your PC, then:  python review_log.py --misses
    """
    if not _authed():
        return jsonify({"error": "set ADMIN_TOKEN in Render, then call /export?token=..."}), 403
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in _turns)
    return Response(body + ("\n" if body else ""),
                    mimetype="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=turns.jsonl"})


@app.route("/chat", methods=["POST"])
def chat():
    p = request.get_json(force=True, silent=True) or {}
    text = (p.get("text") or "").strip()
    session = (p.get("session_id") or "web").strip()
    out = _engine.handle(TENANT, session, text)
    _log_turn(session, text, out)
    return jsonify({"response": out["response"], "source": out["source"]})


PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Receptionist — Training</title>
<style>
  :root { color-scheme: dark; }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{margin:0;min-height:100dvh;display:flex;flex-direction:column;background:#0b1220;color:#e6edf7;
       font:16px/1.5 -apple-system,system-ui,sans-serif}
  header{padding:14px 16px 10px;text-align:center;border-bottom:1px solid #1e2a44}
  header h1{margin:0;font-size:16px}
  .warn{margin:6px auto 0;max-width:520px;padding:6px 10px;border-radius:8px;
        background:#42200f;color:#ffcfa8;font-size:11px;border:1px solid #7c3b12}
  #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
  .msg{max-width:82%;padding:10px 13px;border-radius:16px;white-space:pre-wrap}
  .me{align-self:flex-end;background:#2563eb;border-bottom-right-radius:5px}
  .bot{align-self:flex-start;background:#182338;border-bottom-left-radius:5px}
  .tag{font-size:10px;color:#7d8db0;margin-top:3px}
  footer{padding:12px 16px calc(12px + env(safe-area-inset-bottom));border-top:1px solid #1e2a44;
         display:flex;gap:8px;align-items:center}
  #txt{flex:1;padding:12px;border-radius:22px;border:1px solid #2b3a5c;background:#0f1729;color:#e6edf7;font-size:16px}
  button{border:none;border-radius:50%;width:46px;height:46px;font-size:20px;color:#fff;background:#2563eb}
  #mic.rec{background:#dc2626}
  #send{background:#334155}
  #hint{padding:0 16px 8px;font-size:11px;color:#7d8db0;text-align:center;min-height:15px}
</style></head>
<body>
<header>
  <h1>Dr. Shelby's Office — Training</h1>
  <div class="warn">⚠️ Practice mode. Use FAKE names and dates only — never real patient information.</div>
</header>
<div id="log"></div>
<div id="hint"></div>
<footer>
  <input id="txt" placeholder="Type, or tap the mic…" autocomplete="off">
  <button id="mic">🎤</button>
  <button id="send">➤</button>
</footer>
<script>
const log=document.getElementById('log'), txt=document.getElementById('txt'),
      mic=document.getElementById('mic'), send=document.getElementById('send'),
      hint=document.getElementById('hint');
const session='train-'+Math.random().toString(36).slice(2,10);

function add(t,who,tag){
  const d=document.createElement('div'); d.className='msg '+who; d.textContent=t;
  if(tag){const s=document.createElement('div');s.className='tag';s.textContent=tag;d.appendChild(s);}
  log.appendChild(d); log.scrollTop=log.scrollHeight;
}
// Speak the reply using the BROWSER's voice — no TTS server needed.
function speak(t){
  if(!('speechSynthesis' in window)) return;
  const u=new SpeechSynthesisUtterance(t);
  u.rate=1.0; u.pitch=1.0;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}
async function ask(text){
  if(!text.trim()) return;
  add(text,'me'); txt.value=''; hint.textContent='thinking…';
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, session_id:session})});
    const j=await r.json();
    add(j.response,'bot', j.source?('via '+j.source):'');
    speak(j.response);
    hint.textContent='';
  }catch(e){ hint.textContent='error: '+e.message; }
}
send.onclick=()=>ask(txt.value);
txt.addEventListener('keydown',e=>{ if(e.key==='Enter') ask(txt.value); });

// Speech-to-text, done BY THE BROWSER (Web Speech API).
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec=null, listening=false;
if(!SR){
  mic.disabled=true; mic.style.background='#39456a';
  hint.textContent='Voice input needs Safari or Chrome — you can still type.';
}else{
  rec=new SR(); rec.lang='en-US'; rec.interimResults=false; rec.maxAlternatives=1;
  rec.onresult=e=>{ const t=e.results[0][0].transcript; ask(t); };
  rec.onerror=e=>{ hint.textContent='mic: '+e.error; stop(); };
  rec.onend=()=>stop();
  function stop(){ listening=false; mic.classList.remove('rec'); mic.textContent='🎤'; }
  mic.onclick=()=>{
    if(listening){ rec.stop(); return; }
    speak(' ');                         // unlock audio on iOS via user gesture
    try{ rec.start(); listening=true; mic.classList.add('rec'); mic.textContent='⏹';
         hint.textContent='listening…'; }
    catch(e){ hint.textContent='mic: '+e.message; }
  };
}
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    print(f"[training] serving on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
