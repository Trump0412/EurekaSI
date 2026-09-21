"""Explicit source-empty exclusions, preserving the original group split."""
from decimal import Decimal
import json
from pathlib import Path
import re


def display_size_matches(size, display):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?B)', display)
    if not match:
        raise ValueError('Unknown inventory size syntax')
    number, unit = match.groups()
    scale = Decimal(1024)**['B', 'KB', 'MB', 'GB', 'TB'].index(unit)
    expected = Decimal(number)*scale
    # Byte listings are exact; human-readable larger units are rounded, not an
    # exact checksum/length manifest. Never claim more precision than supplied.
    if unit == 'B':
        return Decimal(size) == expected
    digits = len(number.partition('.')[2])
    tolerance = Decimal('0.5')*(Decimal(10)**(-digits))*scale
    return abs(Decimal(size)-expected) <= tolerance


def audit_source_inventory(inventory_path, media_root):
    inventory = json.loads(Path(inventory_path).read_text())
    files = inventory['files']
    if len(files) != inventory['count']:
        raise ValueError('Inventory count mismatch')
    empty = []
    for name, display in files.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe source filename')
        path = Path(media_root)/name
        if not path.is_file():
            raise ValueError(f'Missing declared file: {name}')
        size = path.stat().st_size
        # A rounded "0.00KB" is not proof of an empty upstream object.
        remote_zero = re.fullmatch(r'0(?:\.0+)?B', display) is not None
        if remote_zero:
            if size != 0:
                raise ValueError('Remote-empty file changed locally; inspect replacement provenance')
            empty.append(name)
        elif size <= 0 or not display_size_matches(size, display):
            raise ValueError(f'Nonempty inventory member incomplete/size mismatch: {name}')
    if not empty:
        raise ValueError('Explicit source-empty recovery requested but no upstream empty files declared')
    return {'inventory_files': len(files), 'nonempty_files': len(files)-len(empty),
            'upstream_empty_files': sorted(empty), 'local_nonempty_sizes_verified': True,
            'size_precision': 'B exact; KB/MB/GB/TB within listed rounding interval, not byte-exact/checksum verification'}


def exclude_source_empty_after_split(rows, empty_names):
    """Called only after explicit opt-in, and after the original group_split."""
    names = set(empty_names)
    retained, excluded = [], []
    for row in rows:
        name = Path(row.get('video_path', '')).name
        if row['source'] == '4drl' and name in names:
            excluded.append({'id': row['id'], 'source': row['source'], 'scene_id': row['scene_id'],
                'source_group': row['source_group'], 'original_split': row['split'], 'video_file': name,
                'reason': 'upstream_empty_source_media', 'remote_bytes': 0, 'local_bytes': 0})
        else:
            retained.append(row)
    return retained, excluded


def reusable_cache(source, directory, count):
    """Select existing cache only when identity matches; never mutate old cache."""
    receipt = Path(directory)/'receipt.json'
    if not receipt.is_file():
        return False
    stat = Path(source).stat()
    value = json.loads(receipt.read_text())
    return value.get('identity') == {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'count': count}
