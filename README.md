# Receptionist — Render training build

**FAKE DATA ONLY.** This is not a HIPAA-enabled deployment. Never enter a real
patient's name, date of birth, or symptoms. Use it to practice and to show
people how the bot answers — nothing else.

## Why this build is different

Render has no GPU, and Whisper + Kokoro + torch won't fit a small instance.
They aren't needed: the **browser** does speech-to-text and text-to-speech
(Web Speech API). Render only runs the engine — Flask + scikit-learn, no ML
models. Fits the free tier.

    browser (STT) --> text --> /chat --> engine (rules + RAG + calendar)
    browser (TTS) <-- text <---------------

Only text crosses the network.

Differences from your local build:
  * LLM: OFF (no GPU on Render)
  * ML intent: OFF
  * Whisper / Kokoro: not used — the browser handles voice
  * Conversation logging: OFF (a training app shouldn't accumulate transcripts)
  * Calendar: writes to /tmp, wiped on restart

## Deploy

1. Push this folder to a GitHub repo.
2. Render -> New -> Web Service -> connect the repo.
3. Render reads `render.yaml`. If setting it manually:
       Build:  pip install -r requirements.txt
       Start:  python app.py
4. Open the URL. Real HTTPS, no certificate warning, works from anywhere.

## Local test

    pip install -r requirements.txt
    python app.py        # http://localhost:10000

## Voice support

Speech input needs Safari (iOS/macOS) or Chrome. Firefox has no speech
recognition — typing still works everywhere. Note that browser speech
recognition sends audio to Apple/Google servers, which is another reason this
build is for fake data only.

## Logs — how to see what the bot is missing

Turns are logged to **stdout**, which Render captures in your (private) dashboard.
Nothing is written to disk: Render's free tier has an ephemeral filesystem, and a
log file holding conversation text is exactly what you don't want on a public URL.

Every turn is redacted twice:
  1. If the engine knows the turn was a patient name or DOB (the booking flow
     just asked for one), the input is dropped entirely — no guessing.
  2. Everything else goes through phi_scrub (phones, emails, dates, IDs, names).

### Read them

**Render dashboard** → your service → **Logs**. Each turn is one JSON line with
what was heard, what was said, and which layer answered.

**/stats** — counts only, no text. Safe to open anywhere:

    https://<your-app>.onrender.com/stats
    -> {"turns": 42, "by_source": {"fsm": 20, "rag": 12, "fallback": 10}, "fallback_rate": 0.238}

**/misses** — the questions it could NOT answer. This echoes back what people
typed, so it's token-protected. In Render: **Environment** → add

    ADMIN_TOKEN = <any long random string>

then:

    https://<your-app>.onrender.com/misses?token=<your token>

Every miss is a gap in your rules or knowledge base. That list is the to-do list.

## Supervisor console

    https://<your-app>.onrender.com/console?token=<ADMIN_TOKEN>

Enter your name in the top-left box. Every live caller appears in the left
column as *Caller 1*, *Caller 2*… (or by caller ID once telephony is wired in).
Click one to read the whole conversation live.

**When the bot can't answer, the call flashes RED and the page beeps.** The
caller hears *"Let me check with my supervisor — one moment"* and waits, rather
than getting the email brush-off. They can also tap 🙋 to ask for a person.

### Two (or more) supervisors

Both consoles see every call. To reply you must **Take this call** first —
that locks it to you. If someone else already has it, your Send is disabled and
their name shows on the call; you can still read it, and you can Take over if
you need to. Without this, two supervisors typing into the same caller would
make them hear two different answers.

### "Listening" to a call

Click **🔊 Listen** and your browser reads that call's transcript aloud as it
happens — so you can follow a conversation without watching it. Each supervisor
can have this on for a *different* call.

Note this is speech synthesis of the transcript, not the caller's real voice.
The caller's audio is recognised in their own browser and never leaves their
device — nothing to stream, which is also the right privacy posture.

### The bot learns from you

Every answer any supervisor types is kept:

    https://<your-app>.onrender.com/kb_suggestions?token=<ADMIN_TOKEN>

That's a `qa_taught.txt` in knowledge-base format. Drop it into
`profiles/default/knowledge/`, redeploy, and the bot answers those questions
itself. **Answer once; the bot knows it forever.**

### Limits

State is in memory. A Render restart clears live calls and any taught answers
you haven't downloaded — grab `qa_taught.txt` after each session. Beyond
training use, this needs a database.
