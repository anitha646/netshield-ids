"""
IDS Flask Backend
Endpoints:
  GET  /                    → dashboard HTML
  GET  /api/status          → model info
  GET  /api/stream          → SSE stream of live predictions
  POST /api/predict         → single prediction from JSON body
  GET  /api/stats           → aggregate stats
  GET  /api/alerts          → last N alerts
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import time
import threading
import queue
from collections import deque, defaultdict
from datetime import datetime

from flask import Flask, Response, request, jsonify, render_template_string
from flask_cors import CORS

from feature_extractor import simulate_flow
from predictor import get_predictor

app = Flask(__name__)
CORS(app)

# ── Shared state ───────────────────────────────────────────────────────────────
MAX_EVENTS    = 200
MAX_ALERTS    = 50
event_log     = deque(maxlen=MAX_EVENTS)
alert_log     = deque(maxlen=MAX_ALERTS)
stats         = defaultdict(int)
stats_lock    = threading.Lock()
sse_clients   = []
sse_lock      = threading.Lock()

def broadcast(data: dict):
    """Push event to all SSE clients"""
    msg = f"data: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

def traffic_generator():
    """Background thread: generate + predict flows continuously"""
    predictor = get_predictor()
    while True:
        try:
            flow = simulate_flow(attack_probability=0.25)
            features = flow.to_model_dict()

            if predictor.is_ready():
                result = predictor.predict(features)
            else:
                result = {'label': 'UNKNOWN', 'confidence': 0, 'severity': 'none',
                          'color': '#888', 'description': 'Model not ready',
                          'is_attack': False, 'class_probabilities': {},
                          'timestamp': datetime.now().isoformat()}

            event = {
                'id': int(time.time() * 1000),
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'src_ip': flow.src_ip,
                'dst_ip': flow.dst_ip,
                'src_port': flow.src_port,
                'dst_port': flow.dst_port,
                'protocol': flow.protocol_type.upper(),
                'service': flow.service,
                'duration': round(flow.duration, 3),
                'src_bytes': int(flow.src_bytes),
                'dst_bytes': int(flow.dst_bytes),
                'serror_rate': flow.serror_rate,
                'count': flow.count,
                **result,
            }

            event_log.appendleft(event)

            with stats_lock:
                stats['total'] += 1
                stats[result['label']] += 1
                if result['is_attack']:
                    stats['attacks'] += 1
                    alert_log.appendleft(event)

            broadcast({'type': 'flow', 'data': event})

            time.sleep(0.6)  # ~1.7 flows/sec

        except Exception as e:
            print(f"[ERROR] Generator: {e}")
            time.sleep(1)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(open(
        os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')
    ).read())

@app.route('/api/status')
def api_status():
    predictor = get_predictor()
    return jsonify({
        'ready': predictor.is_ready(),
        'model': predictor.meta.get('best_model', 'N/A'),
        'accuracy': predictor.meta.get('accuracy', 0),
        'features': len(predictor.feature_names),
    })

@app.route('/api/stream')
def api_stream():
    """Server-Sent Events endpoint"""
    q = queue.Queue(maxsize=50)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    predictor = get_predictor()
    if not predictor.is_ready():
        return jsonify({'error': 'Model not loaded'}), 503
    result = predictor.predict(data)
    return jsonify(result)

@app.route('/api/stats')
def api_stats():
    with stats_lock:
        s = dict(stats)
    total = s.get('total', 1)
    return jsonify({
        **s,
        'attack_rate': round(s.get('attacks', 0) / max(total, 1) * 100, 1),
        'normal_rate': round(s.get('NORMAL', 0) / max(total, 1) * 100, 1),
    })

@app.route('/api/alerts')
def api_alerts():
    n = int(request.args.get('n', 20))
    return jsonify(list(alert_log)[:n])

@app.route('/api/events')
def api_events():
    n = int(request.args.get('n', 50))
    return jsonify(list(event_log)[:n])


# ── Start background thread ────────────────────────────────────────────────────
def start_generator():
    t = threading.Thread(target=traffic_generator, daemon=True)
    t.start()
    print("[IDS] Traffic generator started.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    print("[IDS] Starting Intrusion Detection System...")
    start_generator()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
