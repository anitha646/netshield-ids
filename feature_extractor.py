"""
Network Traffic Feature Extractor
Extracts NSL-KDD-compatible features from simulated or live packets.
For live capture: requires scapy + root privileges.
For demo: uses realistic traffic simulation.
"""

import time
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES   = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'pop3', 'imap', 'https', 'telnet']
FLAGS      = ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'SH', 'OTH']

# ── Traffic profile weights ────────────────────────────────────────────────────
TRAFFIC_PROFILES = {
    'NORMAL': dict(
        protocol_weights=[0.65, 0.25, 0.10],
        flag_weights=[0.85, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],
        src_bytes_mean=1200, src_bytes_std=800,
        dst_bytes_mean=3000, dst_bytes_std=2000,
        duration_mean=2.5, duration_std=3.0,
        serror_rate_max=0.05,
        count_mean=15, count_std=10,
    ),
    'DoS': dict(
        protocol_weights=[0.20, 0.10, 0.70],
        flag_weights=[0.10, 0.60, 0.15, 0.10, 0.03, 0.01, 0.01],
        src_bytes_mean=50, src_bytes_std=30,
        dst_bytes_mean=0, dst_bytes_std=10,
        duration_mean=0.1, duration_std=0.2,
        serror_rate_max=0.98,
        count_mean=490, count_std=20,
    ),
    'Probe': dict(
        protocol_weights=[0.40, 0.40, 0.20],
        flag_weights=[0.50, 0.20, 0.20, 0.05, 0.03, 0.01, 0.01],
        src_bytes_mean=400, src_bytes_std=200,
        dst_bytes_mean=200, dst_bytes_std=150,
        duration_mean=0.5, duration_std=1.0,
        serror_rate_max=0.60,
        count_mean=100, count_std=50,
    ),
    'R2L': dict(
        protocol_weights=[0.90, 0.08, 0.02],
        flag_weights=[0.70, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
        src_bytes_mean=600, src_bytes_std=400,
        dst_bytes_mean=800, dst_bytes_std=600,
        duration_mean=1.5, duration_std=2.0,
        serror_rate_max=0.20,
        count_mean=5, count_std=3,
    ),
    'U2R': dict(
        protocol_weights=[0.95, 0.04, 0.01],
        flag_weights=[0.80, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01],
        src_bytes_mean=1800, src_bytes_std=1200,
        dst_bytes_mean=400, dst_bytes_std=300,
        duration_mean=3.0, duration_std=4.0,
        serror_rate_max=0.15,
        count_mean=3, count_std=2,
    ),
}

@dataclass
class FlowFeatures:
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    count: int
    srv_count: int
    serror_rate: float
    same_srv_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    # Metadata (not used for prediction)
    src_ip: str = '0.0.0.0'
    dst_ip: str = '0.0.0.0'
    src_port: int = 0
    dst_port: int = 80
    timestamp: float = 0.0
    true_label: str = 'UNKNOWN'

    def to_model_dict(self):
        """Return only model-input fields"""
        return {
            'duration': self.duration,
            'protocol_type': self.protocol_type,
            'service': self.service,
            'flag': self.flag,
            'src_bytes': self.src_bytes,
            'dst_bytes': self.dst_bytes,
            'land': self.land,
            'wrong_fragment': self.wrong_fragment,
            'urgent': self.urgent,
            'hot': self.hot,
            'num_failed_logins': self.num_failed_logins,
            'logged_in': self.logged_in,
            'count': self.count,
            'srv_count': self.srv_count,
            'serror_rate': self.serror_rate,
            'same_srv_rate': self.same_srv_rate,
            'dst_host_count': self.dst_host_count,
            'dst_host_srv_count': self.dst_host_srv_count,
        }

def _rand_ip():
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def simulate_flow(attack_probability: float = 0.25) -> FlowFeatures:
    """
    Simulate a realistic network flow.
    attack_probability: chance of generating an attack flow (0–1)
    """
    rng = np.random.default_rng()

    # Decide traffic type
    if rng.random() < attack_probability:
        label = rng.choice(['DoS', 'Probe', 'R2L', 'U2R'], p=[0.50, 0.30, 0.15, 0.05])
    else:
        label = 'NORMAL'

    p = TRAFFIC_PROFILES[label]

    duration    = max(0, rng.normal(p['duration_mean'], p['duration_std']))
    protocol    = rng.choice(PROTOCOLS, p=p['protocol_weights'])
    service     = rng.choice(SERVICES)
    flag        = rng.choice(FLAGS, p=p['flag_weights'])
    src_bytes   = max(0, rng.normal(p['src_bytes_mean'], p['src_bytes_std']))
    dst_bytes   = max(0, rng.normal(p['dst_bytes_mean'], p['dst_bytes_std']))
    serror_rate = min(1.0, max(0, rng.uniform(0, p['serror_rate_max'])))
    count       = max(1, int(rng.normal(p['count_mean'], p['count_std'])))

    return FlowFeatures(
        duration=round(duration, 3),
        protocol_type=protocol,
        service=service,
        flag=flag,
        src_bytes=round(src_bytes),
        dst_bytes=round(dst_bytes),
        land=int(rng.random() < 0.01),
        wrong_fragment=int(rng.poisson(0.1)),
        urgent=int(rng.random() < 0.02),
        hot=int(rng.poisson(1.5 if label != 'NORMAL' else 0.5)),
        num_failed_logins=int(rng.poisson(2 if label in ('R2L','U2R') else 0)),
        logged_in=int(rng.random() < (0.9 if label == 'NORMAL' else 0.3)),
        count=count,
        srv_count=max(1, int(count * rng.uniform(0.3, 1.0))),
        serror_rate=round(serror_rate, 4),
        same_srv_rate=round(rng.uniform(0.7, 1.0) if label == 'NORMAL' else rng.uniform(0, 0.5), 4),
        dst_host_count=int(rng.integers(1, 256)),
        dst_host_srv_count=int(rng.integers(1, 256)),
        src_ip=_rand_ip(),
        dst_ip=_rand_ip(),
        src_port=int(rng.integers(1024, 65535)),
        dst_port=int(rng.choice([80, 443, 22, 21, 25, 53, 3306])),
        timestamp=time.time(),
        true_label=label
    )


# ── Optional: Live capture with scapy ─────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

class LiveCaptureExtractor:
    """
    Captures live packets and extracts basic features.
    Requires root/admin and scapy installed.
    """
    def __init__(self, interface='eth0', timeout=60):
        if not SCAPY_AVAILABLE:
            raise RuntimeError("scapy not installed. Run: pip install scapy")
        self.interface = interface
        self.timeout = timeout
        self.flows = {}

    def _packet_callback(self, pkt):
        if IP not in pkt:
            return
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
        size = len(pkt)
        key = (src, dst, proto)

        if key not in self.flows:
            self.flows[key] = {
                'start': time.time(), 'src_bytes': 0,
                'dst_bytes': 0, 'packets': 0
            }
        self.flows[key]['src_bytes'] += size
        self.flows[key]['packets'] += 1

    def capture(self, count=100):
        sniff(iface=self.interface, prn=self._packet_callback,
              count=count, timeout=self.timeout)
        return self.flows


if __name__ == '__main__':
    print("Simulating 5 network flows:\n")
    for i in range(5):
        flow = simulate_flow(attack_probability=0.5)
        print(f"Flow {i+1}: label={flow.true_label}, protocol={flow.protocol_type}, "
              f"src_bytes={flow.src_bytes:.0f}, serror_rate={flow.serror_rate:.2f}, "
              f"count={flow.count}")
