#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Render-hosted receptionist with a live SUPERVISOR CONSOLE.

*** FAKE DATA ONLY unless you set FAKE_DATA=0 and understand the PHI rules. ***

The bot answers. A human watches. When the bot struggles, the call turns RED on
the supervisor console — and a person can read the whole conversation, pause the
call, and type the real answer straight to the caller.

    caller (browser)  --/chat-->  engine (rules + RAG + calendar)
                      <--reply--
                      --/poll-->  any message a supervisor typed
    supervisor        --/console--> every live call, red when it needs help
                      --/say-----> types into ONE call

Every answer a supervisor types is kept and can be downloaded as ready-made
knowledge-base entries (/kb_suggestions). Answer a question once; the bot knows
it forever.

Local:  pip install -r requirements.txt && python app.py
"""

import os
import json
import time
from datetime import datetime
from collections import Counter, deque

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from engine import Engine, ProfileStore, SessionStore
from phi_scrub import redact_phi

PORT = int(os.environ.get("PORT", "10000"))
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
TENANT = os.environ.get("DEFAULT_TENANT", "default")

# Needed for the supervisor console and every /export-style endpoint.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# FAKE_DATA=1 logs raw text (so you can debug the booking flow). OFF by default:
# this URL is public, and someone will eventually type a real name into it.
FAKE_DATA = os.environ.get("FAKE_DATA", "0") == "1"

# Which answer sources mean "the bot struggled" -> flag the call red.
WEAK_SOURCES = {"fallback"}

app = Flask(__name__)
CORS(app)

_engine = Engine(ProfileStore(PROFILES_DIR), SessionStore())

# ---------------------------------------------------------------------------
# LIVE CALL STATE
#
# One entry per caller. Everything is in memory: Render's free tier has an
# ephemeral filesystem and a single instance, so this is fine for a console you
# watch in real time — but a restart clears it. Download /kb_suggestions after
# each session or the taught answers are lost.
# ---------------------------------------------------------------------------
_calls = {}            # session_id -> call record
_taught = deque(maxlen=500)   # supervisor answers -> future KB entries
_counts = Counter()
_turns = deque(maxlen=5000)
_next_number = [1]


def _call(session_id: str, caller_id: str = "") -> dict:
    """Get (or open) the record for one caller."""
    c = _calls.get(session_id)
    if c is None:
        c = {
            "session": session_id,
            "number": _next_number[0],          # "Caller 1", "Caller 2", ...
            "caller_id": caller_id,             # a phone number, when telephony exists
            "started": datetime.now().isoformat(timespec="seconds"),
            "last": datetime.now().isoformat(timespec="seconds"),
            # The caller page polls every 2s. That poll IS the heartbeat: when it
            # stops, they closed the tab or refreshed — i.e. they hung up.
            "seen": time.time(),
            "turns": [],                        # the whole conversation
            "needs_help": False,                # -> RED on the console
            "reason": "",
            "outbox": [],                       # messages a supervisor typed
            # With two people watching, they must not answer the same caller at
            # once — the caller would hear both. Claiming a call locks it to you.
            "claimed_by": "",
            "claimed_at": "",
        }
        _next_number[0] += 1
        _calls[session_id] = c
    return c


def _scrub(text: str, pii: bool = False) -> str:
    if FAKE_DATA:
        return text
    return "[REDACTED_PATIENT_DETAIL]" if pii else redact_phi(text)


def _authed() -> bool:
    if not ADMIN_TOKEN:
        return False
    tok = request.args.get("token") or (request.get_json(silent=True) or {}).get("token")
    return tok == ADMIN_TOKEN


# ---------------------------------------------------------------------------
# CALLER SIDE
# ---------------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    p = request.get_json(force=True, silent=True) or {}
    text = (p.get("text") or "").strip()
    session = (p.get("session_id") or "web").strip()
    c = _call(session, p.get("caller_id", ""))

    out = _engine.handle(TENANT, session, text)
    source = out.get("source", "?")
    # Stamp liveness AFTER handling, not before: the first request builds the
    # retrieval index and can take seconds, which would leave the heartbeat
    # already stale and get a live caller reaped as a hang-up.
    c["seen"] = time.time()

    heard = _scrub(text, out.get("pii", False))
    said = _scrub(out.get("response", ""))

    c["turns"].append({"ts": datetime.now().isoformat(timespec="seconds"),
                       "who": "caller", "text": heard})
    c["turns"].append({"ts": datetime.now().isoformat(timespec="seconds"),
                       "who": "bot", "text": said, "source": source})
    c["last"] = datetime.now().isoformat(timespec="seconds")

    # The bot couldn't answer -> light this call up RED for the supervisor.
    if source in WEAK_SOURCES:
        c["needs_help"] = True
        c["reason"] = "no answer"

    _counts[source] += 1
    row = {"ts": c["last"], "session": session, "heard": heard, "said": said[:120],
           "source": source, "lane": out.get("lane")}
    _turns.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

    return jsonify({"response": out["response"], "source": source,
                    "needs_help": c["needs_help"]})


# A caller who hasn't polled in this long has hung up (closed or refreshed the
# tab). Their call drops off the console — nobody should be staring at a dead
# call while a live one waits.
HANGUP_AFTER = float(os.environ.get("HANGUP_AFTER", "12"))


@app.route("/poll/<session>")
def poll(session):
    """Caller polls: has a supervisor typed anything to me?
    This doubles as the heartbeat that says they're still on the line."""
    c = _calls.get(session)
    if c:
        c["seen"] = time.time()          # still here
    if not c or not c["outbox"]:
        return jsonify({"messages": []})
    msgs = c["outbox"][:]
    c["outbox"].clear()
    return jsonify({"messages": msgs})


@app.route("/raise_hand", methods=["POST"])
def raise_hand():
    """Caller taps the 'ask a person' button."""
    p = request.get_json(force=True, silent=True) or {}
    session = (p.get("session_id") or "web").strip()
    c = _call(session)
    c["needs_help"] = True
    c["reason"] = "caller asked for a person"
    c["last"] = datetime.now().isoformat(timespec="seconds")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# SUPERVISOR SIDE
# ---------------------------------------------------------------------------
@app.route("/calls")
def calls():
    """Every live call. The console polls this."""
    if not _authed():
        return jsonify({"error": "token required"}), 403

    # Reap hung-up callers. Their transcript is already in the log and in
    # /kb_suggestions, so nothing is lost — it just leaves the console.
    now = time.time()
    for sid in [s for s, c in _calls.items() if now - c["seen"] > HANGUP_AFTER]:
        ended = _calls.pop(sid)
        print(json.dumps({"event": "hung_up", "caller": ended["number"],
                          "turns": len(ended["turns"])}), flush=True)

    rows = sorted(_calls.values(), key=lambda c: c["number"])
    return jsonify({"calls": [{
        "session": c["session"],
        "number": c["number"],
        "caller_id": c["caller_id"],
        "started": c["started"],
        "last": c["last"],
        "needs_help": c["needs_help"],
        "reason": c["reason"],
        "claimed_by": c["claimed_by"],
        "turns": c["turns"][-40:],
    } for c in rows]})


@app.route("/claim", methods=["POST"])
def claim():
    """Take a call so the other supervisor doesn't answer it too.

    Without this, two people watching the same console both type into the same
    caller and the caller hears two different answers.
    """
    if not _authed():
        return jsonify({"error": "token required"}), 403
    p = request.get_json(force=True, silent=True) or {}
    c = _calls.get((p.get("session_id") or "").strip())
    who = (p.get("who") or "someone").strip()
    if not c:
        return jsonify({"error": "unknown call"}), 400
    if c["claimed_by"] and c["claimed_by"] != who and not p.get("steal"):
        return jsonify({"ok": False, "claimed_by": c["claimed_by"]}), 409
    c["claimed_by"] = who
    c["claimed_at"] = datetime.now().isoformat(timespec="seconds")
    return jsonify({"ok": True, "claimed_by": who})


@app.route("/release", methods=["POST"])
def release():
    if not _authed():
        return jsonify({"error": "token required"}), 403
    p = request.get_json(force=True, silent=True) or {}
    c = _calls.get((p.get("session_id") or "").strip())
    if c:
        c["claimed_by"] = ""
    return jsonify({"ok": True})


@app.route("/say", methods=["POST"])
def say():
    """Supervisor types into ONE call. The caller hears it on their next poll,
    and the Q/A is kept so the bot can learn it."""
    if not _authed():
        return jsonify({"error": "token required"}), 403
    p = request.get_json(force=True, silent=True) or {}
    session = (p.get("session_id") or "").strip()
    text = (p.get("text") or "").strip()
    who = (p.get("who") or "supervisor").strip()
    c = _calls.get(session)
    if not c or not text:
        return jsonify({"error": "unknown call or empty message"}), 400

    # Someone else has this call — don't let two supervisors talk over each other.
    if c["claimed_by"] and c["claimed_by"] != who:
        return jsonify({"error": f"{c['claimed_by']} is handling this call",
                        "claimed_by": c["claimed_by"]}), 409

    c["outbox"].append(text)
    c["turns"].append({"ts": datetime.now().isoformat(timespec="seconds"),
                       "who": "supervisor", "text": text, "by": who})
    c["needs_help"] = False
    c["reason"] = ""
    c["last"] = datetime.now().isoformat(timespec="seconds")

    # What was the caller's last question? That + this answer = a KB entry.
    q = next((t["text"] for t in reversed(c["turns"]) if t["who"] == "caller"), "")
    if q:
        _taught.append({"q": q, "a": text,
                        "ts": datetime.now().isoformat(timespec="seconds")})
    print(json.dumps({"event": "supervisor_said", "session": session,
                      "q": q, "a": text}, ensure_ascii=False), flush=True)
    return jsonify({"ok": True})


@app.route("/kb_suggestions")
def kb_suggestions():
    """Supervisor answers, formatted as knowledge-base entries.

    Save into profiles/default/knowledge/ and the bot answers those questions
    itself from then on. This is the point of the whole console.
    """
    if not _authed():
        return jsonify({"error": "token required"}), 403
    if not _taught:
        return Response("# nothing taught yet\n", mimetype="text/plain")
    body = "\n---\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in _taught)
    return Response(body + "\n", mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=qa_taught.txt"})


@app.route("/export")
def export():
    if not _authed():
        return jsonify({"error": "token required"}), 403
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in _turns)
    return Response(body + ("\n" if body else ""), mimetype="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=turns.jsonl"})


@app.route("/stats")
def stats():
    total = sum(_counts.values())
    if not total:
        return jsonify({"turns": 0})
    return jsonify({"turns": total, "by_source": dict(_counts),
                    "fallback_rate": round(_counts["fallback"] / total, 3)})


@app.route("/health")
def health():
    return jsonify({"ok": True, "mode": "training",
                    "logging": "RAW (fake data only)" if FAKE_DATA else "PHI-scrubbed",
                    "live_calls": len(_calls)})


@app.route("/console")
def console():
    if not _authed():
        return Response("<h3>Add ?token=YOUR_ADMIN_TOKEN to the URL</h3>",
                        mimetype="text/html", status=403)
    return Response(CONSOLE_PAGE, mimetype="text/html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    })


CONSOLE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supervisor Console</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0b1220;color:#e6edf7;font:14px/1.5 -apple-system,system-ui,sans-serif}
 header{padding:10px 14px;border-bottom:1px solid #1e2a44;display:flex;align-items:center;gap:10px}
 h1{font-size:15px;margin:0}
 .who{background:#0f1729;border:1px solid #2b3a5c;color:#e6edf7;border-radius:6px;padding:5px 8px;width:120px;font:inherit}
 .count{font-size:12px;color:#7d8db0}
 .alert{margin-left:auto;font-size:12px;color:#f87171;display:none}
 .alert.on{display:block}
 .wrap{display:flex;height:calc(100dvh - 45px)}
 .list{width:300px;border-right:1px solid #1e2a44;overflow-y:auto;flex:none}
 .call{padding:11px 13px;border-bottom:1px solid #172136;cursor:pointer}
 .call:hover{background:#111c30}
 .call.sel{background:#152036;border-left:3px solid #2563eb;padding-left:10px}
 .call.help{background:#3b0f14;border-left:3px solid #dc2626;padding-left:10px;animation:flash 1s infinite}
 .call.mine{border-left:3px solid #16a34a;padding-left:10px}
 @keyframes flash{0%,100%{background:#3b0f14}50%{background:#5c151c}}
 .cname{font-weight:600;display:flex;align-items:center;gap:6px}
 .cmeta{font-size:11px;color:#7d8db0;margin-top:2px}
 .badge{background:#dc2626;color:#fff;font-size:10px;padding:1px 6px;border-radius:9px}
 .lock{background:#334155;color:#cbd5e1;font-size:10px;padding:1px 6px;border-radius:9px}
 .lock.me{background:#166534;color:#dcfce7}
 .pane{flex:1;display:flex;flex-direction:column;min-width:0}
 .bar{padding:8px 14px;border-bottom:1px solid #1e2a44;display:flex;align-items:center;gap:8px;font-size:12px}
 .bar button{background:#334155;color:#fff;border:none;border-radius:6px;padding:5px 11px;font-size:12px}
 .bar button.on{background:#16a34a}
 .bar button.claim{background:#2563eb}
 .log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
 .msg{max-width:78%;padding:8px 12px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
 .caller{align-self:flex-start;background:#1e293b}
 .bot{align-self:flex-end;background:#1e3a5f}
 .sup{align-self:flex-end;background:#166534}
 .src{font-size:10px;color:#7d8db0;margin-top:2px}
 .src.weak{color:#f87171;font-weight:600}
 .compose{border-top:1px solid #1e2a44;padding:11px;display:flex;gap:8px}
 .compose input{flex:1;padding:10px;border-radius:8px;border:1px solid #2b3a5c;background:#0f1729;color:#e6edf7;font:inherit}
 .compose button{background:#16a34a;color:#fff;border:none;border-radius:8px;padding:10px 16px;font-weight:600}
 .compose input:disabled,.compose button:disabled{opacity:.4}
 .empty{color:#7d8db0;text-align:center;padding:50px 20px}
 .foot{padding:7px 13px;border-top:1px solid #1e2a44;font-size:11px;color:#7d8db0}
 a{color:#60a5fa}
</style></head><body>
<header>
  <h1>Console</h1>
  <input class="who" id="who" placeholder="your name">
  <span class="count" id="count">no calls</span>
  <span class="alert" id="alert">🔴 a caller needs help</span>
</header>
<div class="wrap">
  <div class="list" id="list"></div>
  <div class="pane">
    <div class="bar" id="bar">
      <button id="claim" class="claim">Take this call</button>
      <button id="listen">🔊 Listen</button>
      <span id="status" style="color:#7d8db0"></span>
    </div>
    <div class="log" id="log"><div class="empty">select a call</div></div>
    <div class="compose">
      <input id="msg" placeholder="Take the call first, then type…" disabled autocomplete="off">
      <button id="send" disabled>Send</button>
    </div>
    <div class="foot">
      Every answer you type is kept →
      <a id="dl" href="#">qa_taught.txt</a> → drop into
      <code>profiles/default/knowledge/</code> and the bot learns it permanently.
    </div>
  </div>
</div>
<script>
const tok=new URLSearchParams(location.search).get('token');
document.getElementById('dl').href='/kb_suggestions?token='+encodeURIComponent(tok);

const whoBox=document.getElementById('who');
whoBox.value = localStorage.getItem('supname') || '';
whoBox.oninput = ()=> localStorage.setItem('supname', whoBox.value);
const me = ()=> (whoBox.value.trim() || 'supervisor');

let sel=null, calls=[], prevHelp=0;
let listenOn=false, spokenCount={};      // per-call: how many turns we've read aloud

function beep(){
  try{
    const a=new (window.AudioContext||window.webkitAudioContext)();
    const o=a.createOscillator(), g=a.createGain();
    o.connect(g); g.connect(a.destination);
    o.frequency.value=880; g.gain.value=0.08;
    o.start(); setTimeout(()=>{o.stop(); a.close();},220);
  }catch(e){}
}
// "Listening" to a call = the browser reads the transcript aloud. The caller's
// raw audio never leaves their device, so there is nothing to stream — but you
// can still follow a conversation without watching it. Each supervisor can have
// this on for a DIFFERENT call.
function speak(t){
  if(!listenOn || !('speechSynthesis' in window)) return;
  const u=new SpeechSynthesisUtterance(t); u.rate=1.05;
  speechSynthesis.speak(u);
}

async function load(){
  const r=await fetch('/calls?token='+encodeURIComponent(tok));
  if(!r.ok){ document.getElementById('list').innerHTML='<div class="empty">bad token</div>'; return; }
  calls=(await r.json()).calls;

  const help=calls.filter(c=>c.needs_help).length;
  document.getElementById('count').textContent = calls.length ? calls.length+' live' : 'no calls';
  document.getElementById('alert').className = help?'alert on':'alert';
  if(help>prevHelp) beep();
  prevHelp=help;
  document.title = help ? `(${help}) 🔴 Console` : 'Console';

  // Jump to whoever needs help, so you can't sit on the wrong call while a
  // caller waits. Only auto-jumps if you haven't picked one, or the one you're
  // on is fine.
  const needy = calls.find(c=>c.needs_help);
  if(needy){
    const cur = calls.find(c=>c.session===sel);
    if(!cur || !cur.needs_help) sel = needy.session;
  }else if(!sel && calls.length){
    sel = calls[calls.length-1].session;      // otherwise show the newest call
  }

  const list=document.getElementById('list');
  list.innerHTML = calls.length ? '' : '<div class="empty">waiting for callers…</div>';
  for(const c of calls){
    const mine = c.claimed_by===me();
    const d=document.createElement('div');
    d.className='call'+(c.needs_help?' help':'')+(mine?' mine':'')+(sel===c.session?' sel':'');
    const who = c.caller_id || ('Caller '+c.number);
    let tags='';
    if(c.needs_help) tags+='<span class="badge">NEEDS YOU</span>';
    if(c.claimed_by) tags+=`<span class="lock${mine?' me':''}">${mine?'you':c.claimed_by}</span>`;
    d.innerHTML=`<div class="cname">${who}${tags}</div>
                 <div class="cmeta">${c.turns.length} turns · ${c.last.slice(11,16)}${c.reason?' · '+c.reason:''}</div>`;
    d.onclick=()=>{ sel=c.session; render(); };
    list.appendChild(d);
  }
  render();
}

function render(){
  const log=document.getElementById('log');
  const c=calls.find(x=>x.session===sel);
  const claimBtn=document.getElementById('claim');
  const listenBtn=document.getElementById('listen');
  const status=document.getElementById('status');
  const msg=document.getElementById('msg'), send=document.getElementById('send');

  if(!c){
    log.innerHTML='<div class="empty">select a call</div>';
    msg.disabled=send.disabled=true; status.textContent='';
    return;
  }
  const mine = c.claimed_by===me();
  const takenByOther = c.claimed_by && !mine;

  claimBtn.textContent = mine ? 'Release' : (takenByOther ? 'Take over' : 'Take this call');
  claimBtn.onclick = async ()=>{
    if(mine){
      await fetch('/release?token='+encodeURIComponent(tok),{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({session_id:c.session})});
    }else{
      if(takenByOther && !confirm(c.claimed_by+' has this call. Take it anyway?')) return;
      await fetch('/claim?token='+encodeURIComponent(tok),{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({session_id:c.session, who:me(), steal:takenByOther})});
    }
    load();
  };
  listenBtn.className = listenOn ? 'on' : '';
  listenBtn.onclick = ()=>{ listenOn=!listenOn; if(!listenOn) speechSynthesis.cancel(); render(); };

  status.textContent = takenByOther ? (c.claimed_by+' is handling this call')
                    : (mine ? 'you have this call' : 'take the call to reply');
  msg.disabled = send.disabled = !mine;
  msg.placeholder = mine ? 'Type what the caller should hear…' : 'Take the call first…';

  log.innerHTML='';
  for(const t of c.turns){
    const d=document.createElement('div');
    d.className='msg '+(t.who==='caller'?'caller':(t.who==='supervisor'?'sup':'bot'));
    d.textContent=t.text;
    const s=document.createElement('div');
    s.className='src'+(t.source==='fallback'?' weak':'');
    s.textContent = t.who==='supervisor' ? ('— '+(t.by||'supervisor')) : (t.source||'');
    if(s.textContent) d.appendChild(s);
    log.appendChild(d);
  }
  log.scrollTop=log.scrollHeight;

  // read aloud only what's NEW on the selected call
  const seen = spokenCount[c.session] || 0;
  if(listenOn && c.turns.length > seen){
    for(const t of c.turns.slice(seen)){
      if(t.who==='caller') speak('Caller says: '+t.text);
      else if(t.who==='bot') speak('Bot says: '+t.text);
    }
  }
  spokenCount[c.session]=c.turns.length;
}

async function send(){
  const box=document.getElementById('msg');
  if(!sel || !box.value.trim()) return;
  const text=box.value; box.value='';
  const r=await fetch('/say?token='+encodeURIComponent(tok),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:sel, text, who:me()})});
  if(r.status===409){
    const j=await r.json();
    alert(j.error||'someone else has this call');
  }
  load();
}
document.getElementById('send').onclick=send;
document.getElementById('msg').addEventListener('keydown',e=>{ if(e.key==='Enter') send(); });

load(); setInterval(load, 2000);
</script></body></html>"""


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
  #human{background:#a16207}
  .waiting{align-self:flex-start;background:#2a2412;border:1px dashed #7c5e12;color:#fde68a;
           padding:10px 13px;border-radius:16px;font-size:14px}
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
  <button id="human" title="Ask a person">🙋</button>
</footer>
<script>
const log=document.getElementById('log'), txt=document.getElementById('txt'),
      mic=document.getElementById('mic'), send=document.getElementById('send'),
      hint=document.getElementById('hint');
// A page load is a new caller — and a REFRESH is a hang-up followed by a new
// call, which is exactly what it looks like to the office. The old call stops
// polling, the server notices, and it drops off the console.
const session = 'call-' + Math.random().toString(36).slice(2,8);

function add(t,who,tag){
  const d=document.createElement('div'); d.className='msg '+who; d.textContent=t;
  if(tag){const s=document.createElement('div');s.className='tag';s.textContent=tag;d.appendChild(s);}
  log.appendChild(d); log.scrollTop=log.scrollHeight;
}
function speak(t){
  if(!('speechSynthesis' in window)) return;
  const u=new SpeechSynthesisUtterance(t); u.rate=1.0;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

let waiting=null;
function showWaiting(){
  if(waiting) return;
  waiting=document.createElement('div');
  waiting.className='waiting';
  waiting.textContent='⏳ checking with my supervisor…';
  log.appendChild(waiting); log.scrollTop=log.scrollHeight;
}
function clearWaiting(){ if(waiting){ waiting.remove(); waiting=null; } }

async function ask(text){
  if(!text.trim()) return;
  add(text,'me'); txt.value=''; hint.textContent='thinking…';
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, session_id:session})});
    const j=await r.json();
    if(j.source==='fallback'){
      // The bot doesn't know. Don't brush the caller off — get a person.
      const line="Let me check with my supervisor — one moment.";
      add(line,'bot'); speak(line); showWaiting();
      hint.textContent='a person is looking at this…';
    }else{
      add(j.response,'bot', j.source?('via '+j.source):'');
      speak(j.response); hint.textContent='';
    }
  }catch(e){ hint.textContent='error: '+e.message; }
}
send.onclick=()=>ask(txt.value);
txt.addEventListener('keydown',e=>{ if(e.key==='Enter') ask(txt.value); });

// Ask for a human deliberately.
document.getElementById('human').onclick=async()=>{
  await fetch('/raise_hand',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:session})});
  const line="Let me get someone for you — one moment.";
  add(line,'bot'); speak(line); showWaiting();
  hint.textContent='a person is looking at this…';
};

// A supervisor can type into this call at ANY time — poll for it.
setInterval(async()=>{
  try{
    const r=await fetch('/poll/'+session);
    const j=await r.json();
    for(const m of (j.messages||[])){
      clearWaiting();
      add(m,'bot','from a person');
      speak(m);
      hint.textContent='';
    }
  }catch(e){}
}, 2000);

// browser speech-to-text
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec=null, listening=false;
if(!SR){
  mic.disabled=true; mic.style.background='#39456a';
  hint.textContent='Voice needs Safari or Chrome — you can still type.';
}else{
  rec=new SR(); rec.lang='en-US'; rec.interimResults=false;
  rec.onresult=e=>ask(e.results[0][0].transcript);
  rec.onerror=e=>{ hint.textContent='mic: '+e.error; stopRec(); };
  rec.onend=()=>stopRec();
  function stopRec(){ listening=false; mic.classList.remove('rec'); mic.textContent='🎤'; }
  mic.onclick=()=>{
    if(listening){ rec.stop(); return; }
    speak(' ');
    try{ rec.start(); listening=true; mic.classList.add('rec'); mic.textContent='⏹';
         hint.textContent='listening…'; }
    catch(e){ hint.textContent='mic: '+e.message; }
  };
}
</script></body></html>"""


SUPERVISOR_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supervisor</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0b1220;color:#e6edf7;font:15px/1.5 -apple-system,system-ui,sans-serif;padding:18px}
 h1{font-size:17px;margin:0 0 4px}
 .sub{color:#7d8db0;font-size:12px;margin-bottom:16px}
 .card{background:#152036;border:1px solid #24334f;border-radius:12px;padding:14px;margin-bottom:12px}
 .q{font-weight:600;margin-bottom:4px}
 .meta{font-size:11px;color:#7d8db0;margin-bottom:10px}
 textarea{width:100%;min-height:70px;background:#0f1729;color:#e6edf7;border:1px solid #2b3a5c;
          border-radius:8px;padding:10px;font:inherit;resize:vertical;box-sizing:border-box}
 button{margin-top:8px;background:#16a34a;color:#fff;border:none;border-radius:8px;
        padding:9px 18px;font-weight:600}
 .empty{color:#7d8db0;text-align:center;padding:40px 0}
 .foot{margin-top:22px;font-size:12px;color:#7d8db0;border-top:1px solid #24334f;padding-top:12px}
 a{color:#60a5fa}
</style></head><body>
<h1>Waiting for a human</h1>
<div class="sub">The caller is still on the line. Type the real answer and it goes straight back to them.</div>
<div id="list"><div class="empty">nothing waiting…</div></div>
<div class="foot">
  Every answer you type is kept. Download them all as knowledge-base entries:
  <a id="dl" href="#">qa_taught.txt</a> — drop that into
  <code>profiles/default/knowledge/</code> and the bot will answer it itself next time.
</div>
<script>
const tok=new URLSearchParams(location.search).get('token');
document.getElementById('dl').href='/kb_suggestions?token='+encodeURIComponent(tok);
async function load(){
  const r=await fetch('/pending?token='+encodeURIComponent(tok));
  const j=await r.json();
  const list=document.getElementById('list');
  if(!j.pending || !j.pending.length){ list.innerHTML='<div class="empty">nothing waiting…</div>'; return; }
  list.innerHTML='';
  for(const p of j.pending){
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<div class="q">${p.question.replace(/</g,'&lt;')}</div>
                 <div class="meta">${p.ts} · session ${p.session}</div>
                 <textarea placeholder="Type the answer the caller should hear..."></textarea>
                 <button>Send to caller</button>`;
    const ta=d.querySelector('textarea'), btn=d.querySelector('button');
    btn.onclick=async()=>{
      if(!ta.value.trim()) return;
      btn.disabled=true; btn.textContent='sending…';
      await fetch('/answer?token='+encodeURIComponent(tok),{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:p.id, answer:ta.value})});
      load();
    };
    list.appendChild(d);
  }
}
load(); setInterval(load, 3000);
</script></body></html>"""


@app.route("/")
def index():
    # No caching. Safari in particular will hold on to an old copy of this page,
    # and a stale page polls the WRONG endpoint — the caller sits on "let me
    # check with my supervisor" forever while the supervisor's answer goes
    # nowhere. Always serve fresh.
    return Response(PAGE, mimetype="text/html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })


if __name__ == "__main__":
    print(f"[training] serving on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
