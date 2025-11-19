import os, random, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify, make_response
from openai import OpenAI
from wordfreq import top_n_list, zipf_frequency

# ---------- config ----------
VLLM_BASE   = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")  # vLLM or Ollama /v1
VLLM_KEY    = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL       = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # or "tinyllama" for Ollama
LIMIT_VOCAB = int(os.getenv("LIMIT_VOCAB", "50000"))
MIN_ZIPF    = float(os.getenv("MIN_ZIPF", "3.4"))  # filter very rare words
SEED_LEN    = 5                                     # EXACTLY 5 letters

client = OpenAI(base_url=VLLM_BASE, api_key=VLLM_KEY, timeout=5.0)

# ---------- vocab / anagram index (5-letter only) ----------
def signature(w: str) -> str:
    return "".join(sorted(w))

def build_five_index(limit_vocab=LIMIT_VOCAB, min_zipf=MIN_ZIPF):
    words = [w.lower() for w in top_n_list("en", n=limit_vocab) if w.isalpha()]
    five  = [w for w in words if len(w) == SEED_LEN and zipf_frequency(w, "en") >= min_zipf]
    idx: Dict[str, List[str]] = {}
    for w in five:
        idx.setdefault(signature(w), []).append(w)
    for k in idx:
        idx[k] = sorted(list(set(idx[k])))
    return five, idx

FIVE_WORDS, ANAGRAM5_INDEX = build_five_index()

def five_letter_anagrams(letters_sorted: str) -> List[str]:
    return ANAGRAM5_INDEX.get(letters_sorted, [])

def scramble(letters_sorted: str) -> str:
    arr = list(letters_sorted)
    random.shuffle(arr)
    return "".join(arr)

# ---------- in-memory sessions ----------
@dataclass
class GameState:
    topic: str
    letters: str      # sorted 5 letters
    scrambled: str
    targets: List[str]  # all valid 5-letter anagrams (>=1)
    found: List[str] = field(default_factory=list)

SESSIONS: Dict[str, GameState] = {}

# ---------- seed pickers ----------
def _fallback_topic_word() -> Tuple[str, str]:
    topic = random.choice(["animals", "music", "space", "sports", "food"])
    # any 5-letter word is fine (even if only 1 anagram)
    return (topic, random.choice(FIVE_WORDS)) if FIVE_WORDS else (topic, "tiger")

def ask_tinyllama_topic_and_word() -> Tuple[str, str]:
    """Ask LLM for a 5-letter (topic, word); fallback locally if needed."""
    if os.getenv("DISABLE_LLM", "0") == "1" or not FIVE_WORDS:
        return _fallback_topic_word()

    sys = ("You are a word game generator. Respond ONLY in JSON with keys: topic, word. "
           f"topic: a broad fun category. word: a common lowercase English word of EXACTLY {SEED_LEN} letters (a–z only).")
    usr = f'Example: {{"topic":"animals","word":"tiger"}}'

    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sys},
                          {"role": "user",   "content": usr}],
                temperature=0.8, max_tokens=60
            )
            import json, re
            txt = r.choices[0].message.content.strip()
            blob = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            topic = str(blob["topic"]).strip()
            word  = str(blob["word"]).strip().lower()
        except Exception:
            continue
        if word.isalpha() and len(word) == SEED_LEN and zipf_frequency(word, "en") >= MIN_ZIPF:
            return topic, word

    return _fallback_topic_word()

# ---------- app & routes ----------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/")
def home():
    html = """<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Unscramble the Word by Jonathan Lee</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .fade-in { animation: fade .35s ease-in-out both; }
    @keyframes fade { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;} }
  </style>
</head>
<body class="bg-gradient-to-b from-slate-50 to to-slate-100 text-slate-800 min-h-screen">
  <div class="max-w-3xl mx-auto p-6">
    <header class="mb-6">
      <h1 class="text-3xl font-bold tracking-tight">Unscramble the Word by Jonathan Lee</h1>
      <p class="text-sm text-slate-600">Unscramble the word by finding all 5-letter words that can be formed from the given letters. You can start a new round by pressing the "0" key.</p>
    </header>

    <section class="bg-white rounded-2xl shadow border border-slate-200 p-5 fade-in">
      <div class="flex flex-wrap items-center gap-3 mb-4">
        <button id="newBtn" class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white transition">New Round</button>
        <button id="shuffleBtn" class="px-4 py-2 rounded-xl bg-white border hover:bg-slate-50 transition">Shuffle Letters</button>
        <span id="topicPill" class="ml-auto text-xs px-2 py-1 rounded-full bg-slate-100 border text-slate-600 hidden"></span>
      </div>

      <div class="mb-2">
        <div class="text-xs text-slate-500">Scrambled</div>
        <div id="scrambled" class="font-mono text-3xl tracking-wider select-none py-1">—</div>
      </div>

      <div class="flex items-center gap-2 mt-4">
        <input id="guessInput" class="flex-1 border rounded-xl px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500"
               placeholder="Type a 5-letter guess and press Enter…" autocomplete="off" maxlength="5">
        <button id="guessBtn" class="px-4 py-2 rounded-xl bg-slate-800 hover:bg-black text-white transition">Submit</button>
      </div>

      <div id="status" class="text-sm text-slate-500 mt-2 h-5"></div>

      <div class="mt-4">
        <div class="flex items-center justify-between text-xs text-slate-500 mb-1">
          <span>Progress</span>
          <span id="progText">0 / 0</span>
        </div>
        <div class="h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <div id="progBar" class="h-full bg-indigo-600 w-0 transition-all duration-300"></div>
        </div>
      </div>

      <div class="mt-4">
        <div class="text-xs text-slate-500 mb-1">Found words</div>
        <div id="foundWrap" class="flex flex-wrap gap-2"></div>
      </div>

      <div class="mt-5 text-xs text-slate-400">Tip: press <kbd class="px-1.5 py-0.5 border rounded">0</kbd> to start a new round.</div>
    </section>
  </div>

<script>
const newBtn = document.getElementById('newBtn');
const shuffleBtn = document.getElementById('shuffleBtn');
const guessBtn = document.getElementById('guessBtn');
const guessInput = document.getElementById('guessInput');
const scrambledEl = document.getElementById('scrambled');
const topicPill = document.getElementById('topicPill');
const statusEl = document.getElementById('status');
const foundWrap = document.getElementById('foundWrap');
const progBar = document.getElementById('progBar');
const progText = document.getElementById('progText');

let sid = ''; let letters = ''; let found = []; let total = 0;

const setStatus = (msg, tone='muted') => {
  const colors = { muted:'text-slate-500', ok:'text-emerald-600', warn:'text-amber-600', err:'text-rose-600' };
  statusEl.className = 'text-sm mt-2 h-5 ' + (colors[tone] || colors.muted);
  statusEl.textContent = msg || '';
};

const renderFound = () => {
  foundWrap.innerHTML = '';
  found.forEach(w => {
    const chip = document.createElement('span');
    chip.className = 'px-2 py-1 rounded-full bg-slate-100 border text-sm';
    chip.textContent = w;
    foundWrap.appendChild(chip);
  });
  progText.textContent = `${found.length} / ${total}`;
  const pct = total ? Math.round((found.length / total)*100) : 0;
  progBar.style.width = pct + '%';
};

const shuffle = (s) => s.split('').sort(()=>Math.random()-0.5).join('');

async function newRound() {
  setStatus('Creating round…', 'muted');
  newBtn.disabled = true; guessBtn.disabled = true;
  guessInput.value = ''; guessInput.blur();
  try {
    const r = await fetch('/api/game/new', { method:'POST' });
    if (!r.ok) throw new Error('Server error');
    const j = await r.json();
    sid = j.session_id; letters = j.letters; found = []; total = j.answer_count || 0;
    topicPill.textContent = `Topic: ${j.topic}`;
    topicPill.classList.remove('hidden');
    scrambledEl.textContent = j.scrambled || shuffle(letters);
    renderFound();
    setStatus('Round ready. Type a 5-letter guess!', 'ok');
    guessInput.focus();
  } catch (e) {
    setStatus('Could not start round. Try again.', 'err');
  } finally {
    newBtn.disabled = false; guessBtn.disabled = false;
  }
}

async function submitGuess() {
  const g = (guessInput.value || '').trim().toLowerCase();
  if (!g) return;
  if (!sid) { setStatus('Start a new round first.', 'warn'); return; }
  if (g.length !== 5) { setStatus('Guess must be exactly 5 letters.', 'warn'); return; }

  guessBtn.disabled = true; setStatus('Checking…', 'muted');
  try {
    const r = await fetch('/api/game/guess', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: sid, guess: g })
    });
    const j = await r.json();
    if (r.ok && j.correct) {
      found = j.found || found;
      renderFound();
      setStatus('Nice! Correct.', 'ok');
      guessInput.value = '';
      if (j.complete) setStatus('🎉 All 5-letter anagrams found! Press “New Round”.', 'ok');
    } else if (r.ok && !j.correct) {
      setStatus('Not one of the 5-letter words for these letters.', 'warn');
    } else {
      setStatus(j.error || 'Error', 'err');
    }
  } catch(e) {
    setStatus('Network error.', 'err');
  } finally {
    guessBtn.disabled = false; guessInput.focus();
  }
}

newBtn.onclick = newRound;
guessBtn.onclick = submitGuess;
guessInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitGuess(); });
shuffleBtn.onclick = () => {
  if (!letters) return;
  scrambledEl.textContent = shuffle(letters);
  scrambledEl.classList.add('fade-in');
  setTimeout(()=>scrambledEl.classList.remove('fade-in'), 350);
};

// Hotkey: "0" (zero) to start a new round — only when NOT typing
window.addEventListener('keydown', (e) => {
  const el = document.activeElement;
  const isTyping = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
  if (isTyping) return;
  const key = (e.key || '').toLowerCase();
  if (key === '0' || e.code === 'Numpad0') { e.preventDefault(); newRound(); }
});

newRound();
</script>
</body></html>"""
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

@app.post("/api/game/new")
def new_game():
    try:
        topic, word = ask_tinyllama_topic_and_word()
    except Exception:
        topic, word = _fallback_topic_word()

    # Enforce exactly 5 letters; build 5-letter anagram list (>=1)
    if len(word) != SEED_LEN:
        topic, word = _fallback_topic_word()

    letters = "".join(sorted(word))
    targets = five_letter_anagrams(letters)
    if not targets:  # ultra-edge case
        topic, word = _fallback_topic_word()
        letters = "".join(sorted(word))
        targets = five_letter_anagrams(letters)

    scram = scramble(letters)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = GameState(topic=topic, letters=letters, scrambled=scram, targets=targets)

    return jsonify({
        "session_id": sid,
        "topic": topic,
        "scrambled": scram,
        "letters": letters,            # 5 letters
        "answer_count": len(targets)   # how many 5-letter words exist for this signature
    })

@app.post("/api/game/guess")
def guess():
    data = request.get_json(force=True)
    sid   = data.get("session_id")
    g     = str(data.get("guess", "")).lower().strip()
    if not sid or sid not in SESSIONS:
        return jsonify({"error": "invalid session"}), 400
    gs = SESSIONS[sid]

    ok = (len(g) == SEED_LEN and g in gs.targets and signature(g) == gs.letters)
    if ok and g not in gs.found:
        gs.found.append(g)

    return jsonify({
        "correct": ok,
        "found": gs.found,
        "remaining": len(gs.targets) - len(gs.found),
        "complete": len(gs.found) == len(gs.targets)
    })

@app.get("/api/game/state")
def state():
    sid = request.args.get("session_id", "")
    if sid not in SESSIONS:
        return jsonify({"error": "invalid session"}), 400
    gs = SESSIONS[sid]
    return jsonify({
        "topic": gs.topic,
        "scrambled": gs.scrambled,
        "letters": gs.letters,
        "found": gs.found,
        "total": len(gs.targets)
    })