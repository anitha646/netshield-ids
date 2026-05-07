"""
IDS ML Model Trainer
Dataset: NSL-KDD (auto-downloaded) or synthetic fallback
Models: Random Forest + XGBoost ensemble
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed. Using Random Forest only.")

# ── NSL-KDD column names ───────────────────────────────────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# ── Attack type mapping ────────────────────────────────────────────────────────
ATTACK_MAP = {
    'normal': 'NORMAL',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'xlock': 'R2L',
    'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

BINARY_LABEL_MAP = {'NORMAL': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}

def generate_synthetic_data(n=10000):
    """Fallback: generate synthetic NSL-KDD-like data for demo"""
    print("[INFO] Generating synthetic NSL-KDD-like dataset...")
    np.random.seed(42)

    normal = pd.DataFrame({
        'duration': np.random.exponential(2, n//2),
        'protocol_type': np.random.choice(['tcp','udp','icmp'], n//2, p=[0.6,0.3,0.1]),
        'service': np.random.choice(['http','ftp','smtp','ssh','dns'], n//2),
        'flag': np.random.choice(['SF','S0','REJ','RSTO'], n//2, p=[0.8,0.1,0.05,0.05]),
        'src_bytes': np.random.lognormal(6, 2, n//2),
        'dst_bytes': np.random.lognormal(6, 2, n//2),
        'land': np.zeros(n//2, dtype=int),
        'wrong_fragment': np.zeros(n//2, dtype=int),
        'urgent': np.zeros(n//2, dtype=int),
        'hot': np.random.poisson(1, n//2),
        'num_failed_logins': np.zeros(n//2, dtype=int),
        'logged_in': np.ones(n//2, dtype=int),
        'count': np.random.randint(1, 50, n//2),
        'srv_count': np.random.randint(1, 50, n//2),
        'serror_rate': np.random.uniform(0, 0.1, n//2),
        'same_srv_rate': np.random.uniform(0.7, 1.0, n//2),
        'dst_host_count': np.random.randint(100, 256, n//2),
        'dst_host_srv_count': np.random.randint(50, 256, n//2),
        'label': 'NORMAL'
    })

    attacks = []
    attack_types = [('DoS', n//8), ('Probe', n//8), ('R2L', n//16), ('U2R', n//16)]
    for atype, cnt in attack_types:
        a = pd.DataFrame({
            'duration': np.random.exponential(0.5, cnt),
            'protocol_type': np.random.choice(['tcp','udp','icmp'], cnt),
            'service': np.random.choice(['http','ftp','smtp','ssh','dns'], cnt),
            'flag': np.random.choice(['S0','REJ','RSTO','SF'], cnt, p=[0.5,0.2,0.2,0.1]),
            'src_bytes': np.random.lognormal(3, 3, cnt) if atype == 'DoS' else np.random.lognormal(5, 2, cnt),
            'dst_bytes': np.random.lognormal(2, 2, cnt),
            'land': np.random.randint(0, 2, cnt),
            'wrong_fragment': np.random.randint(0, 3, cnt),
            'urgent': np.random.randint(0, 2, cnt),
            'hot': np.random.poisson(5, cnt),
            'num_failed_logins': np.random.randint(0, 5, cnt),
            'logged_in': np.random.randint(0, 2, cnt),
            'count': np.random.randint(200, 512, cnt) if atype == 'DoS' else np.random.randint(1, 100, cnt),
            'srv_count': np.random.randint(200, 512, cnt) if atype == 'DoS' else np.random.randint(1, 100, cnt),
            'serror_rate': np.random.uniform(0.5, 1.0, cnt) if atype in ['DoS','Probe'] else np.random.uniform(0, 0.3, cnt),
            'same_srv_rate': np.random.uniform(0, 0.3, cnt),
            'dst_host_count': np.random.randint(1, 256, cnt),
            'dst_host_srv_count': np.random.randint(1, 100, cnt),
            'label': atype
        })
        attacks.append(a)

    df = pd.concat([normal] + attacks, ignore_index=True).sample(frac=1, random_state=42)
    return df

def load_nsl_kdd(data_dir='data'):
    """Try to load NSL-KDD; fall back to synthetic"""
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    if os.path.exists(train_path):
        print(f"[INFO] Loading NSL-KDD from {train_path}")
        df = pd.read_csv(train_path, names=COLUMNS)
        df['label'] = df['label'].str.strip().str.lower().map(ATTACK_MAP).fillna('NORMAL')
        return df
    else:
        print(f"[WARN] NSL-KDD not found at {train_path}. Using synthetic data.")
        print("[HINT] Download from: https://www.unb.ca/cic/datasets/nsl.html")
        return generate_synthetic_data()

def preprocess(df):
    """Encode categoricals, scale numerics"""
    CATEGORICAL = ['protocol_type', 'service', 'flag']
    NUMERIC_FEATURES = [
        'duration','src_bytes','dst_bytes','land','wrong_fragment','urgent',
        'hot','num_failed_logins','logged_in','count','srv_count',
        'serror_rate','same_srv_rate','dst_host_count','dst_host_srv_count'
    ]

    encoders = {}
    for col in CATEGORICAL:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    available_features = [c for c in NUMERIC_FEATURES + CATEGORICAL if c in df.columns]

    X = df[available_features].copy()
    y = df['label'].map(BINARY_LABEL_MAP).fillna(0).astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler, encoders, available_features

def train_and_save(data_dir='data', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_nsl_kdd(data_dir)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")

    X, y, scaler, encoders, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"[RF] Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_preds, target_names=list(BINARY_LABEL_MAP.keys())))
    results['random_forest'] = rf_acc

    # ── XGBoost ────────────────────────────────────────────────────────────────
    best_model = rf
    best_name = 'random_forest'

    if HAS_XGB:
        print("[INFO] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1, random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        print(f"[XGB] Accuracy: {xgb_acc:.4f}")
        print(classification_report(y_test, xgb_preds, target_names=list(BINARY_LABEL_MAP.keys())))
        results['xgboost'] = xgb_acc

        if xgb_acc > rf_acc:
            best_model = xgb_model
            best_name = 'xgboost'

    print(f"\n[INFO] Best model: {best_name} — saving artifacts...")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    with open(os.path.join(model_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(BINARY_LABEL_MAP, f)

    import json
    meta = {
        'best_model': best_name,
        'accuracy': results.get(best_name, 0),
        'features': feature_names,
        'label_map': BINARY_LABEL_MAP,
        'results': results
    }
    with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] All artifacts saved to ./{model_dir}/")
    print(f"[DONE] Final accuracy: {meta['accuracy']:.4f}")
    return meta

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    train_and_save(data_dir=data_dir)
