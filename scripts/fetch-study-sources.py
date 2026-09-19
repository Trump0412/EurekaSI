"""Fetch exact external renderer/scorer sources; do not vendor upstream methods."""
import argparse
from pathlib import Path
import subprocess
import json

SOURCES={
 'draw_marker.py':('Li-Hao-yuan/GeoThinker','e9b3786ab5847618b406e41a661e2106f2aa82af','src/qwen_vl/data/draw_marker.py'),
 'revsi_utils.py':('EvolvingLMMs-Lab/lmms-eval','1cd474f858a3055407d44a4b823e3d6d3299bbe7','lmms_eval/tasks/revsi/utils.py'),
 'revsi_template.yaml':('EvolvingLMMs-Lab/lmms-eval','1cd474f858a3055407d44a4b823e3d6d3299bbe7','lmms_eval/tasks/revsi/_default_template_yaml'),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    folder=Path(a.root)/'sources';folder.mkdir(parents=True,exist_ok=True)
    for name,(repo,rev,path) in SOURCES.items():
        dest=folder/name
        if dest.exists():continue
        url=f'https://raw.githubusercontent.com/{repo}/{rev}/{path}'
        tmp=dest.with_suffix('.part')
        subprocess.run(['curl','-fLsS','--connect-timeout','15','--max-time','120','--retry','3','-o',str(tmp),url],check=True)
        tmp.replace(dest)
    (folder/'provenance.json').write_text(json.dumps(SOURCES,indent=2))


if __name__=='__main__':main()
