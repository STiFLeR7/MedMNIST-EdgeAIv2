import json
from argparse import Namespace

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return Namespace(**cfg)
