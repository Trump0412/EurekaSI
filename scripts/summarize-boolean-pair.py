"""Preserve a paired output-validity diagnostic alongside existing RL/SFT reports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.boolean_metrics import paired_format_analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',required=True);parser.add_argument('--after',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    report=paired_format_analysis(read(args.before),read(args.after))
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve the previous diagnostic output')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
