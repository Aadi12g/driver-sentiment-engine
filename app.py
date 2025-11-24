import os, json, time, datetime, jwt, hashlib, threading, logging, uuid
from queue import Queue
from logging.handlers import RotatingFileHandler
from flask import (Flask, request, make_response, redirect, url_for,render_template_string, jsonify)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Logging Setup
if not os.path.exists("logs"):
    os.makedirs("logs")

log_handler = RotatingFileHandler("logs/app.log", maxBytes=500000, backupCount=5)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[log_handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)
# Config

app = Flask(__name__)
SECRET = "super_secret_jwt_key_change_me"   
GRAPH_PATH = os.path.join("static", "graph.png")
SCHEMA_PATH = "form_schema.json"
DATA_PATH = "data.json"
FEEDBACK_HISTORY_PATH = "feedback_history.json"
os.makedirs("static", exist_ok=True)

# This ensures tokens from a previous process run become invalid.
INSTANCE_ID = str(uuid.uuid4())
logger.info("Instance ID: %s", INSTANCE_ID)

plt.figure(figsize=(9, 2.2), dpi=150)
plt.ylim(0, 5)
plt.yticks([0, 1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(GRAPH_PATH)
plt.close()
# Sentiment Analyzer
class SentimentAnalyzer:
    POS = ["good","great","excellent","friendly","clean","comfortable","safe","nice","helpful","on time","ontime","punctual","polite","smooth","awesome","best","amazing", "😊","😊","😀","👍","🙌"]
    NEG = ["bad","rude","late","unsafe","dirty","angry","reckless","dangerous","uncomfortable","delay","hate","terrible","😡","😠","😡","👎","😞","😒","😢","😤","😣"]

    def classify(self, text):
        t = (text or "").lower()
        if any(w in t for w in self.POS):
            return 5
        if any(w in t for w in self.NEG):
            return 1
        return 3
# Engine (in-memory)
class SentimentEngine:
    def __init__(self):
        self.data = {}               
        self.queue = Queue()
        self.analyzer = SentimentAnalyzer()
        self.threshold = 2.5
        self.alert_history = []
        self.feedback_history = []
        if os.path.exists(FEEDBACK_HISTORY_PATH):
          try:
            self.feedback_history = json.load(open(FEEDBACK_HISTORY_PATH))
          except:
            self.feedback_history = []
        self.processed_uids = set()
        self.lock = threading.Lock()
        self._agg_interval = 60
        self._agg_thread = None
        self._agg_running = False

    def save(self):
        try:
            json.dump(self.data, open(DATA_PATH, "w"))
            json.dump(self.feedback_history, open(FEEDBACK_HISTORY_PATH, "w"))

            logger.debug("Saved data.json")
        except Exception as e:
            logger.warning("Save failed: %s", e)

    def _make_fallback_uid(self, ftype, fid, text):
        s = f"{ftype}|{fid}|{text}|{int(time.time())}"
        return hashlib.sha256(s.encode()).hexdigest()

    def submit(self, ftype, fid, text, raw_fields=None):
        self.queue.put((ftype, fid, text, raw_fields or {}))
        logger.debug("ENQUEUED feedback: %s %s (queue=%d)", ftype, fid, self.queue.qsize())

    def process(self):
        while not self.queue.empty():
            try:
                ftype, fid, text, raw_fields = self.queue.get()
                uid = None
                if isinstance(raw_fields, dict):
                    uid = raw_fields.get("uid")
                if not uid:
                    uid = self._make_fallback_uid(ftype, fid, text)

                with self.lock:
                    if uid in self.processed_uids:
                        logger.info("DUPLICATE feedback uid=%s skipped", uid)
                        continue
                    self.processed_uids.add(uid)

                score = self.analyzer.classify(text)

                with self.lock:
                    self.feedback_history.append({
                        "uid": uid,
                        "type": ftype,
                        "id": fid,
                        "text": text,
                        "fields": raw_fields,
                        "score": score,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    if len(self.feedback_history) > 500:
                        self.feedback_history = self.feedback_history[-500:]

                self.data.setdefault(ftype, {})
                d = self.data[ftype].get(fid, {"score": 0.0, "count": 0})
                d["score"] = (d["score"] * d["count"] + score) / (d["count"] + 1)
                d["count"] += 1
                self.data[ftype][fid] = d

                logger.info("PROCESSED: %s %s score=%s", ftype, fid, score)

                if d["score"] < self.threshold:
                    with self.lock:
                        self.alert_history.append({
                            "type": ftype,
                            "id": fid,
                            "score": round(d["score"], 2),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        if len(self.alert_history) > 200:
                            self.alert_history = self.alert_history[-200:]
                    logger.warning("ALERT: %s %s avg=%s", ftype, fid, d["score"])
            except Exception as e:
                logger.error("Error processing queue item: %s", e, exc_info=True)
        self.save()

    def periodic_aggregate(self):
        logger.info("Periodic aggregate: rebuilding aggregates from in-memory history")
        with self.lock:
            history = list(self.feedback_history)
        new_data = {}
        for f in history:
            ftype = f["type"]; fid = f["id"]; score = float(f["score"])
            new_data.setdefault(ftype, {})
            d = new_data[ftype].get(fid, {"score": 0.0, "count": 0})
            d["score"] = (d["score"] * d["count"] + score) / (d["count"] + 1)
            d["count"] += 1
            new_data[ftype][fid] = d
        with self.lock:
            self.data = new_data
        self.save()
        logger.info("Periodic aggregate completed")

    def start_periodic_aggregation(self, interval=None):
        if interval is not None:
            self._agg_interval = interval
        if self._agg_running:
            return
        self._agg_running = True
        def run_loop():
            while self._agg_running:
                try:
                    self.periodic_aggregate()
                except Exception as e:
                    logger.error("Periodic agg error: %s", e, exc_info=True)
                time.sleep(self._agg_interval)
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        self._agg_thread = t
        logger.info("Started periodic aggregation thread interval=%s", self._agg_interval)

    def stop_periodic_aggregation(self):
        self._agg_running = False
        if self._agg_thread:
            self._agg_thread.join(timeout=1)
            self._agg_thread = None
            logger.info("Stopped periodic aggregation")

engine = SentimentEngine()
engine.start_periodic_aggregation(interval=60)
# Users + Schema
USERS = {
    "admin": {"password": "1234", "role": "admin"},
    "aadi": {"password": "1111", "role": "user"},
    "aadi1": {"password": "2222", "role": "user"}
}

DEFAULT_SCHEMA = [
    {"name": "type", "label": "Type", "kind": "select",
     "options": ["driver", "trip", "app", "marshal"], "required": True},
    {"name": "id", "label": "Entity ID", "kind": "text", "required": True},
    {"name": "text", "label": "Feedback", "kind": "textarea", "required": True}
]

def load_schema():
    if os.path.exists(SCHEMA_PATH):
        try:
            return json.load(open(SCHEMA_PATH))
        except:
            return DEFAULT_SCHEMA.copy()
    else:
        json.dump(DEFAULT_SCHEMA, open(SCHEMA_PATH, "w"))
        return DEFAULT_SCHEMA.copy()

def save_schema(schema):
    json.dump(schema, open(SCHEMA_PATH, "w"))
    logger.info("Form schema saved")
# JWT helpers (including instance id)
def gen_jwt(username, role):
    payload = {
        "user": username,
        "role": role,
        "exp": (datetime.datetime.utcnow() + datetime.timedelta(hours=6)).timestamp(),
        "inst": INSTANCE_ID
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    return token if isinstance(token, str) else token.decode()

def decode_token(token):
    try:
        data = jwt.decode(token, SECRET, algorithms=["HS256"])
        if data.get("inst") != INSTANCE_ID:
            logger.info("Token rejected due to differing instance id")
            return None
        exp = data.get("exp")
        if exp and time.time() > float(exp):
            return None
        return data
    except Exception as e:
        logger.debug("decode_token failed: %s", e)
        return None

def get_user_auth(req):
    token = req.cookies.get("user_token")
    if not token:
        return None
    info = decode_token(token)
    if not info:
        return None
    if info.get("role") != "user":
        return None
    return {"user": info.get("user"), "role": "user"}

def get_admin_auth(req):
    token = req.cookies.get("admin_token")
    if not token:
        return None
    info = decode_token(token)
    if not info:
        return None
    if info.get("role") != "admin":
        return None
    return {"user": info.get("user"), "role": "admin"}

def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# Templates
HEAD = """
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
 body{background:#f4f6fb;}
 .card-soft{border-radius:12px;box-shadow:0 8px 28px rgba(0,0,0,0.06);}
 .brand{color:#0d6efd;font-weight:600;}
 .muted{color:#6c757d;}
 pre.raw{white-space:pre-wrap;word-wrap:break-word;max-height:140px;overflow:auto;}
</style>
"""
LOGIN_HTML = HEAD + """
<div class="container py-5">
  <div class="row justify-content-center">
    <div class="col-md-6">
      <div class="card card-soft p-4">
        <h3 class="brand">Driver Sentiment</h3>
        <form method="POST" autocomplete="off">
          <input class="form-control mb-2" name="u" placeholder="username">
          <input class="form-control mb-3" name="p" placeholder="password" type="password">
          <div class="d-grid gap-2">
            <button class="btn btn-primary">Login</button>
          </div>
        </form>
        <p class="text-danger mt-3">{{msg}}</p>
        {% if logged_as %}
          <div class="mt-3">
            <p class="muted small">You are already logged in as <strong>{{logged_as.user}}</strong> ({{logged_as.role}}).</p>
            <a class="btn btn-outline-primary btn-sm" href="{{ continue_url }}">Continue to dashboard</a>
            <a class="btn btn-outline-danger btn-sm" href="/logout?clear=1">Logout (clear session)</a>
            <p class="muted small mt-2">To sign in as a different role in this browser tab, submit credentials below — this will set the new token for that role.</p>
          </div>
        {% endif %}
      </div>
    </div>
  </div>
</div>
"""
USER_FEEDBACK_HTML = HEAD + """
<nav class="navbar bg-white shadow-sm">
  <div class="container">
    <div><span class="brand">Driver Sentiment</span> &nbsp;<span class="muted">| User Feedback</span></div>
    <div>
      <span class="me-3 muted">Logged in as: <strong>{{user}}</strong></span>
      <a class="btn btn-outline-danger btn-sm" href="/logout">Logout</a>
    </div>
  </div>
</nav>
<div class="container py-4">
  <div class="card card-soft p-4 mx-auto" style="max-width:620px">
    <h5>User Feedback</h5>
    <form id="uform" autocomplete="off">
      {% for f in schema %}
        <div class="mb-2">
          <label class="form-label">{{f.label}}</label>
          {% if f.kind == 'text' %}
            <input class="form-control" name="{{f.name}}" {% if f.required %}required{% endif %}>
          {% elif f.kind == 'textarea' %}
            <textarea class="form-control" name="{{f.name}}" rows="3" {% if f.required %}required{% endif %}></textarea>
          {% elif f.kind == 'select' %}
            <select class="form-select" name="{{f.name}}" {% if f.required %}required{% endif %}>
              {% for o in f.options %}<option value="{{o}}">{{o}}</option>{% endfor %}
            </select>
          {% endif %}
        </div>
      {% endfor %}
      <div class="d-grid gap-2"><button class="btn btn-success" type="submit">Submit</button></div>
    </form>
    <div id="msg" class="mt-3"></div>
  </div>
</div>

<script>
async function whoami(){
  const r = await fetch('/api/whoami_user');
  if (r.status !== 200) location.href = '/';
}
whoami();

document.getElementById('uform').addEventListener('submit', async function(e){
  e.preventDefault();
  const form = e.target;
  const data = {};
  for (const el of form.elements){
    if (!el.name) continue;
    data[el.name] = el.value;
  }
  try {
    const res = await fetch('/user/submit', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    if (res.status === 200){
      document.getElementById('msg').innerHTML = '<div class="alert alert-success">Feedback submitted successfully.</div>';
      form.reset();
      return;
    }
    const txt = await res.text();
    document.getElementById('msg').innerHTML = '<div class="alert alert-danger">'+ (txt || 'Submit failed') +'</div>';
  } catch (err) {
    document.getElementById('msg').innerHTML = '<div class="alert alert-danger">Network error, please try again.</div>';
  }
});

// prevent back button from exposing login
window.history.pushState(null, "", window.location.href);
window.onpopstate = function () { window.history.pushState(null, "", window.location.href); }
</script>
"""
ADMIN_DASH_HTML = HEAD + """
<nav class="navbar bg-white shadow-sm">
  <div class="container">
    <div><span class="brand">Driver Sentiment</span> &nbsp;<span class="muted">| Admin Dashboard</span></div>
    <div>
      <span class="me-3 muted">Logged in as: <strong>{{user}}</strong></span>
      <a class="btn btn-outline-primary btn-sm me-2" href="/admin/feedback-editor">Edit Form</a>
      <a class="btn btn-outline-secondary btn-sm me-2" href="/admin/run_aggregate">Run Agg</a>
      <a class="btn btn-outline-danger btn-sm" href="/logout">Logout</a>
    </div>
  </div>
</nav>

<div class="container py-3">
  <div class="card card-soft p-3 mb-3">
    <h5>Overview</h5>
    <img id="chart" src="{{graph}}" class="img-fluid">
  </div>

  <div class="row">
    <div class="col-lg-8">
      <div id="entities" class="card card-soft p-3 mb-3">
        <h5>Entities & Scores</h5>
        {% if not data %}<p class="muted">No feedback yet.</p>{% else %}
          {% for t, ids in data.items() %}
            <h6 class="mt-3">{{t.title()}}</h6>
            {% for i,v in ids.items() %}
              <div class="d-flex justify-content-between border-bottom py-2">
                <div><strong>{{i}}</strong></div>
                <div>Score: <b>{{'%.2f'|format(v.score)}}</b> | Count: {{v.count}}</div>
              </div>
            {% endfor %}
          {% endfor %}
        {% endif %}
      </div>

      <div id="recentFeedback" class="card card-soft p-3 mb-3">
        <h5>Recent Feedbacks (latest first)</h5>
        {% if not feedbacks %}<p class="muted small">No feedback yet.</p>{% else %}
          {% for f in feedbacks %}
            <div class="border-bottom py-2 small">
              <b>{{ f.type.title() }} {{ f.id }}</b> — Score: {{ f.score }} <span class="muted">• {{ f.timestamp }}</span>
              <div class="mt-1"><pre class="raw">{{ f.text }}</pre></div>
            </div>
          {% endfor %}
        {% endif %}
      </div>
    </div>

    <div class="col-lg-4">
      <div id="recentAlerts" class="card card-soft p-3 mb-3">
        <h6>Recent Alerts</h6>
        {% if not alerts %}<p class="muted small">No alerts yet.</p>{% else %}
          {% for a in alerts %}
            <div class="border-bottom py-2 small">
              <b>{{ a.type.title() }} {{ a.id }}</b><br>Score: {{ a.score }} <span class="muted">• {{ a.timestamp }}</span>
            </div>
          {% endfor %}
        {% endif %}
      </div>

      <div class="card card-soft p-3">
        <h6>Notes</h6>
        <p class="muted small mb-0">Dashboard polls server every 2s to show live updates.</p>
      </div>
    </div>
  </div>
</div>

<script>
async function refresh(){
  const r = await fetch('/api/state_admin');
  if (r.status === 401) { location.href = '/'; return; }
  const j = await r.json();
  document.getElementById('chart').src = '/static/graph.png?ts=' + new Date().getTime();

  // entities
  const ent = document.getElementById('entities');
  let html = '<h5>Entities & Scores</h5>';
  if (!j.data || Object.keys(j.data).length === 0) html += '<p class="muted">No feedback yet.</p>';
  else {
    for (const t of Object.keys(j.data)){
      html += `<h6 class="mt-3">${t.charAt(0).toUpperCase()+t.slice(1)}</h6>`;
      for (const id of Object.keys(j.data[t])){
        const v = j.data[t][id];
        html += `<div class="d-flex justify-content-between border-bottom py-2"><div><strong>${id}</strong></div><div>Score: <b>${v.score.toFixed(2)}</b> | Count: ${v.count}</div></div>`;
      }
    }
  }
  ent.innerHTML = html;

  // feedbacks
  const fb = document.getElementById('recentFeedback');
  let fbHtml = '<h5>Recent Feedbacks (latest first)</h5>';
  if (!j.feedbacks || j.feedbacks.length===0) fbHtml += '<p class="muted small">No feedback yet.</p>';
  else {
    for (const f of j.feedbacks){
      fbHtml += `<div class="border-bottom py-2 small"><b>${f.type.toUpperCase()} ${f.id}</b> — Score: ${f.score} <span class="muted">• ${f.timestamp}</span><div class="mt-1"><pre class="raw">${f.text}</pre></div></div>`;
    }
  }
  fb.innerHTML = fbHtml;

  // alerts
  const al = document.getElementById('recentAlerts');
  let alHtml = '<h6>Recent Alerts</h6>';
  if (!j.alerts || j.alerts.length===0) alHtml += '<p class="muted small">No alerts yet.</p>';
  else {
    for (const a of j.alerts) alHtml += `<div class="border-bottom py-2 small"><b>${a.type.toUpperCase()} ${a.id}</b><br>Score: ${a.score} <span class="muted">• ${a.timestamp}</span></div>`;
  }
  al.innerHTML = alHtml;
}

refresh();
setInterval(refresh, 2000);

// prevent back button from exposing login
window.history.pushState(null, "", window.location.href);
window.onpopstate = function () { window.history.pushState(null, "", window.location.href); }
</script>
"""
EDITOR_HTML = HEAD + """
<nav class="navbar bg-white shadow-sm">
  <div class="container">
    <div><span class="brand">Driver Sentiment</span> &nbsp;<span class="muted">| Feedback Form Editor</span></div>
    <div>
      <a class="btn btn-outline-primary btn-sm me-2" href="/admin/dashboard">Dashboard</a>
      <a class="btn btn-outline-danger btn-sm" href="/logout">Logout</a>
    </div>
  </div>
</nav>

<div class="container py-4">
  <div class="card card-soft p-3 mb-3">
    <h5>Form Schema</h5>
    <p class="muted small">Add / rename / delete fields. Supported kinds: text, textarea, select (comma-separated options).</p>

    <div id="schemaList">
      {% for f in schema %}
        <div class="border-bottom py-2">
          <form class="row g-2 editRow" data-name="{{f.name}}">
            <div class="col-md-3"><input class="form-control" name="name" value="{{f.name}}" readonly></div>
            <div class="col-md-3"><input class="form-control" name="label" value="{{f.label}}"></div>
            <div class="col-md-2">
              <select class="form-select" name="kind">
                <option value="text" {% if f.kind=='text' %}selected{% endif %}>text</option>
                <option value="textarea" {% if f.kind=='textarea' %}selected{% endif %}>textarea</option>
                <option value="select" {% if f.kind=='select' %}selected{% endif %}>select</option>
              </select>
            </div>
            <div class="col-md-3"><input class="form-control" name="options" placeholder="opt1,opt2" value="{{ ','.join(f.options) if f.kind=='select' and f.options else '' }}"></div>
            <div class="col-md-1">
              <button class="btn btn-danger btn-sm deleteBtn" type="button">Del</button>
            </div>
          </form>
        </div>
      {% endfor %}
    </div>

    <hr>
    <h6>Add new field</h6>
    <form id="addForm" class="row g-2">
      <div class="col-md-3"><input class="form-control" name="name" placeholder="field_name"></div>
      <div class="col-md-3"><input class="form-control" name="label" placeholder="Label"></div>
      <div class="col-md-2">
        <select class="form-select" name="kind"><option value="text">text</option><option value="textarea">textarea</option><option value="select">select</option></select>
      </div>
      <div class="col-md-3"><input class="form-control" name="options" placeholder="opt1,opt2 (for select)"></div>
      <div class="col-md-1"><button class="btn btn-primary" id="addBtn" type="submit">Add</button></div>
    </form>

    <div class="mt-3">
      <button id="saveBtn" class="btn btn-success">Save Schema</button>
      <span id="saveMsg" class="ms-3"></span>
    </div>
  </div>
</div>

<script>
async function saveSchema(schema){
  const res = await fetch('/admin/save_schema', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(schema) });
  if (res.status===200) {
    document.getElementById('saveMsg').innerText = 'Saved';
    setTimeout(()=>document.getElementById('saveMsg').innerText='',1500);
  } else {
    document.getElementById('saveMsg').innerText = 'Save failed';
  }
}

document.getElementById('addForm').addEventListener('submit', function(e){
  e.preventDefault();
  const f = e.target;
  const name = (f.name.value||'').trim();
  const label = (f.label.value||'').trim();
  const kind = f.kind.value;
  const opts = (f.options.value||'').split(',').map(s=>s.trim()).filter(Boolean);
  if (!name || !label){ alert('name and label required'); return; }
  const html = `<div class="border-bottom py-2"><form class="row g-2 editRow" data-name="${name}"><div class="col-md-3"><input class="form-control" name="name" value="${name}" readonly></div><div class="col-md-3"><input class="form-control" name="label" value="${label}"></div><div class="col-md-2"><select class="form-select" name="kind"><option value="text"${kind=='text'?' selected':''}>text</option><option value="textarea"${kind=='textarea'?' selected':''}>textarea</option><option value="select"${kind=='select'?' selected':''}>select</option></select></div><div class="col-md-3"><input class="form-control" name="options" value="${opts.join(',')}"></div><div class="col-md-1"><button class="btn btn-danger btn-sm deleteBtn" type="button">Del</button></div></form></div>`;
  document.getElementById('schemaList').insertAdjacentHTML('beforeend', html);
  f.reset();
});

document.getElementById('schemaList').addEventListener('click', function(e){
  if (e.target.classList.contains('deleteBtn')){
    const row = e.target.closest('.border-bottom');
    row.remove();
  }
});

document.getElementById('saveBtn').addEventListener('click', function(){
  const rows = document.querySelectorAll('#schemaList .editRow');
  const sch = [];
  for (const r of rows){
    const name = r.querySelector('[name=name]').value.trim();
    const label = r.querySelector('[name=label]').value.trim();
    const kind = r.querySelector('[name=kind]').value;
    const opts = (r.querySelector('[name=options]').value||'').split(',').map(s=>s.trim()).filter(Boolean);
    const item = {"name":name,"label":label,"kind":kind,"required":true};
    if (kind==='select') item["options"]=opts;
    sch.push(item);
  }
  saveSchema(sch);
});
</script>
"""
# API endpoints
@app.route("/api/whoami_user")
def api_whoami_user():
    info = get_user_auth(request)
    if not info:
        return ("", 401)
    return jsonify({"user": info["user"], "role": info["role"]})

@app.route("/api/state_admin")
def api_state_admin():
    info = get_admin_auth(request)
    if not info:
        return ("", 401)

    logger.info("ADMIN DASHBOARD POLL")

    labels, scores = [], []
    for t, ids in engine.data.items():
        for i, v in ids.items():
            labels.append(f"{t}-{i}")
            scores.append(float(v["score"]))

    if labels:
        plt.figure(figsize=(9, 2.2), dpi=150)
        plt.bar(range(len(scores)), scores, color="#1f77b4")
        plt.xticks(range(len(scores)), labels, rotation=40, fontsize=8)
        plt.ylim(0, 5)
        plt.yticks([0, 1, 2, 3, 4, 5])
        plt.ylabel("Avg Sentiment")
        plt.tight_layout()
        plt.savefig(GRAPH_PATH)
        plt.close()

    data = {}
    for t, ids in engine.data.items():
        data[t] = {k: {"score": float(v["score"]), "count": int(v["count"])} for k, v in ids.items()}

    alerts = list(engine.alert_history[-10:])[::-1]
    feedbacks = list(engine.feedback_history[-10:])[::-1]

    return jsonify({"data": data, "alerts": alerts, "feedbacks": feedbacks})
# Routes
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = (request.form.get("u") or "").strip()
        p = (request.form.get("p") or "").strip()
        if u in USERS and USERS[u]["password"] == p:
            role = USERS[u]["role"]
            token = gen_jwt(u, role)
            logger.info("LOGIN: user=%s role=%s", u, role)
            if role == "user":
                resp = make_response(redirect(url_for("user_feedback")))
                resp.set_cookie("user_token", token, httponly=True, samesite="Lax")
                return no_cache(resp)
            else:
                resp = make_response(redirect(url_for("admin_dashboard")))
                resp.set_cookie("admin_token", token, httponly=True, samesite="Lax")
                return no_cache(resp)
        logger.warning("LOGIN FAILED: user=%s", u)
        logged = get_admin_auth(request) or get_user_auth(request)
        continue_url = url_for("admin_dashboard") if get_admin_auth(request) else (url_for("user_feedback") if get_user_auth(request) else "/")
        return no_cache(make_response(render_template_string(LOGIN_HTML, msg="Invalid credentials", logged_as=logged, continue_url=continue_url)))
    else:

        logged = get_admin_auth(request) or get_user_auth(request)
        continue_url = url_for("admin_dashboard") if get_admin_auth(request) else (url_for("user_feedback") if get_user_auth(request) else "/")
        return no_cache(make_response(render_template_string(LOGIN_HTML, msg="", logged_as=logged, continue_url=continue_url)))

@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("login")))
    # Ensure both are cleared
    resp.set_cookie("user_token", "", expires=0)
    resp.set_cookie("admin_token", "", expires=0)
    logger.info("LOGOUT (cleared tokens)")
    return no_cache(resp)
# User endpoints
@app.route("/user/feedback", methods=["GET"])
def user_feedback():
    info = get_user_auth(request)
    if not info:
        return no_cache(make_response(redirect(url_for("login"))))
    schema = load_schema()
    return no_cache(make_response(render_template_string(USER_FEEDBACK_HTML, schema=schema, user=info["user"])))

@app.route("/user/submit", methods=["POST"])
def user_submit():
    info = get_user_auth(request)
    if not info:
        return ("Not authenticated as user", 401)
    payload = request.get_json() or {}
    schema = load_schema()
    raw_fields = {}
    missing = []
    main_text = ""
    for f in schema:
        name = f["name"]; req = f.get("required", True)
        val = (payload.get(name) or "").strip()
        if req and not val:
            missing.append(name)
        raw_fields[name] = val
        if name == "text":
            main_text = val
    if missing:
        logger.warning("FEEDBACK MISSING: %s", missing)
        return ("Missing fields: " + ", ".join(missing), 400)
    if payload.get("uid"):
        raw_fields["uid"] = payload.get("uid")
    ftype = raw_fields.get("type") or "driver"
    fid = raw_fields.get("id") or ("unknown-" + str(int(time.time())))
    text_for_sent = main_text or " ".join([v for v in raw_fields.values() if v])
    logger.info("FEEDBACK RECEIVED: user=%s type=%s id=%s", info['user'], ftype, fid)
    engine.submit(ftype, fid, text_for_sent, raw_fields)
    logger.info("QUEUE SIZE: %d", engine.queue.qsize())
    engine.process()
    return ("OK", 200)

# Admin endpoints
@app.route("/admin/dashboard", methods=["GET"])
def admin_dashboard():
    info = get_admin_auth(request)
    if not info:
        return no_cache(make_response(redirect(url_for("login"))))
    
    labels, scores = [], []
    for t, ids in engine.data.items():
        for i, v in ids.items():
            labels.append(f"{t}-{i}")
            scores.append(float(v["score"]))
    if labels:
        plt.figure(figsize=(9, 2.2), dpi=150)
        plt.bar(range(len(scores)), scores, color="#1f77b4")
        plt.xticks(range(len(scores)), labels, rotation=40, fontsize=8)
        plt.ylim(0, 5)
        plt.yticks([0, 1, 2, 3, 4, 5])
        plt.ylabel("Avg Sentiment")
        plt.tight_layout()
        plt.savefig(GRAPH_PATH)
        plt.close()
    else:
        if not os.path.exists(GRAPH_PATH):
            plt.figure(figsize=(9, 2.2), dpi=150)
            plt.ylim(0, 5)
            plt.yticks([0, 1, 2, 3, 4, 5])
            plt.tight_layout()
            plt.savefig(GRAPH_PATH)
            plt.close()
    wrapped = {
        t: {
            k: type("X", (object,), {"score": v["score"], "count": v["count"]})
            for k, v in ids.items()
        }
        for t, ids in engine.data.items()
    }
    alerts = engine.alert_history[-10:][::-1] if engine.alert_history else []
    feedbacks = engine.feedback_history[-10:][::-1] if engine.feedback_history else []
    return no_cache(make_response(
        render_template_string(
            ADMIN_DASH_HTML,
            data=wrapped,
            graph="/static/graph.png",
            alerts=alerts,
            feedbacks=feedbacks,
            user=info["user"]
        )
    ))

@app.route("/admin/feedback-editor", methods=["GET"])
def admin_editor():
    info = get_admin_auth(request)
    if not info:
        return no_cache(make_response(redirect(url_for("login"))))
    schema = load_schema()
    return no_cache(make_response(render_template_string(EDITOR_HTML, schema=schema)))

@app.route("/admin/save_schema", methods=["POST"])
def admin_save_schema():
    info = get_admin_auth(request)
    if not info:
        return ("", 401)
    payload = request.get_json() or []
    names = set()
    for f in payload:
        if not f.get("name") or not f.get("label") or not f.get("kind"):
            return ("Invalid schema", 400)
        if f["name"] in names:
            return ("Duplicate field names", 400)
        names.add(f["name"])
        if f["kind"] == "select" and not isinstance(f.get("options", []), list):
            return ("Invalid options", 400)
    save_schema(payload)
    logger.info("ADMIN updated schema")
    return ("OK", 200)

@app.route("/admin/run_aggregate", methods=["GET"])
def admin_run_aggregate():
    info = get_admin_auth(request)
    if not info:
        return no_cache(make_response(redirect(url_for("login"))))
    try:
        engine.periodic_aggregate()
        return no_cache(make_response(redirect(url_for("admin_dashboard"))))
    except Exception as e:
        logger.error("Manual aggregate failed: %s", e)
        return (f"Aggregation failed: {e}", 500)

if __name__ == "__main__":
    logger.info("Starting app (pure in-memory, INSTANCE_ID=%s)", INSTANCE_ID)
    app.run(debug=True)
