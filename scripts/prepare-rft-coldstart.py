"""Prepare or finalize a locked cold-start plan. Never starts annotation/training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.rft_coldstart import read, prepare, finalize

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    (finalize if a.finalize else prepare)(read(a.plan))
