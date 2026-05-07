"""
IDS Predictor
Loads trained model artifacts and runs real-time inference on flow features.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

LABEL_NAMES = {0: 'NORMAL', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
SEVERITY    = {'NORMAL': 'none', 'DoS': 'critical', 'Probe': 'medium', 'R2L': 'high', 'U2R': 'critical'}
COLORS      = {'NORMAL': '#22c55e', 'DoS': '#ef4444', 'Probe': '#f59e0b', 'R2L': '#f97316', 'U2R': '#dc2626'}
DESCRIPTIONS = {
    'NORMAL': 'Legitimate network traffic — no threat detected.',
    'DoS':    'Denial of Service attack — flooding target with traffic to exhaust resources.',
    'Probe':  'Network probe/scan — attacker mapping hosts, open ports, or vulnerabilities.',
    'R2L':    'Remote-to-Local attack — unauthorized access attempt from remote machine.',
    'U2R':    'User-to-Root attack — privilege escalation attempt to gain root access.',
}

class IDSPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.meta = {}
        self._load()

    def _load(self):
        try:
            with open(os.path.join(self.model_dir, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.model_dir, 'encoders.pkl'), 'rb') as f:
                self.encoders = pickle.load(f)
            with open(os.path.join(self.model_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
            with open(os.path.join(self.model_dir, 'meta.json'), 'r') as f:
                self.meta = json.load(f)
            print(f"[IDS] Model loaded: {self.meta.get('best_model','?')} | accuracy={self.meta.get('accuracy',0):.4f}")
        except FileNotFoundError:
            print("[IDS] No model found. Run train_model.py first.")
            self.model = None

    def is_ready(self):
        return self.model is not None

    def predict(self, flow_dict: dict) -> dict:
        """
        flow_dict: raw feature dict from FlowFeatures.to_model_dict()
        Returns prediction dict with label, confidence, severity, etc.
        """
        if not self.is_ready():
            return {'error': 'Model not loaded'}

        df = pd.DataFrame([flow_dict])

        # Encode categoricals
        for col, le in self.encoders.items():
            if col in df.columns:
                val = df[col].iloc[0]
                if val not in le.classes_:
                    val = le.classes_[0]
                df[col] = le.transform([val])

        # Align to trained feature set
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        # Scale
        X = self.scaler.transform(df)

        # Predict
        pred_class = int(self.model.predict(X)[0])
        label = LABEL_NAMES.get(pred_class, 'UNKNOWN')

        # Confidence
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            class_probs = {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba) if i in LABEL_NAMES}
        else:
            confidence = 1.0
            class_probs = {label: 1.0}

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'severity': SEVERITY[label],
            'color': COLORS[label],
            'description': DESCRIPTIONS[label],
            'class_probabilities': class_probs,
            'timestamp': datetime.now().isoformat(),
            'is_attack': label != 'NORMAL',
        }

    def predict_batch(self, flow_dicts: list) -> list:
        return [self.predict(f) for f in flow_dicts]


# Singleton
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = IDSPredictor()
    return _predictor


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import simulate_flow

    pred = get_predictor()
    if pred.is_ready():
        print("\nRunning 10 predictions:\n")
        correct = 0
        for _ in range(10):
            flow = simulate_flow(attack_probability=0.4)
            result = pred.predict(flow.to_model_dict())
            match = '✓' if result['label'] == flow.true_label else '✗'
            print(f"  True: {flow.true_label:8s} | Pred: {result['label']:8s} | "
                  f"Conf: {result['confidence']:.2f} | {match}")
            if result['label'] == flow.true_label:
                correct += 1
        print(f"\nSample accuracy: {correct}/10")
