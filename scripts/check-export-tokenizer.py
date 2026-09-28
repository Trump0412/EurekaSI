"""Compare saved tokenizer JSON semantics without importing model libraries."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', required=True)
parser.add_argument('--export', required=True)
args = parser.parse_args()
base = json.loads((Path(args.base) / 'tokenizer.json').read_text())
exported = json.loads((Path(args.export) / 'tokenizer.json').read_text())
result = {'equal': base == exported, 'different_fields': [key for key in base.keys() | exported.keys() if base.get(key) != exported.get(key)]}
result['model_differences'] = {key: {'base_type': type(base['model'].get(key)).__name__,
    'export_type': type(exported['model'].get(key)).__name__,
    'base_preview': repr(base['model'].get(key))[:120], 'export_preview': repr(exported['model'].get(key))[:120]}
    for key in base['model'].keys() | exported['model'].keys() if base['model'].get(key) != exported['model'].get(key)}
from tokenizers import Tokenizer
result['runtime_semantics_equal'] = json.loads(Tokenizer.from_file(str(Path(args.base) / 'tokenizer.json')).to_str()) == json.loads(Tokenizer.from_file(str(Path(args.export) / 'tokenizer.json')).to_str())
print(json.dumps(result))
if not result['runtime_semantics_equal']:
    raise SystemExit(1)
