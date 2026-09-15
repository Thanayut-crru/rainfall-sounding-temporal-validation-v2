"""Refresh descriptive distance columns only, without rerunning any experiment.

Default is dry-run. Use --apply after reviewing the report. Files are preflighted
before any replacement. All non-distance cells are compared as CSV strings (no
float parsing or round-trip rounding), and original bytes are backed up first.
RAINFALL_PROJECT_DIR / RAINFALL_OUTPUT_DIR select the same paths as other scripts.
This utility does not rewrite old fit-provenance hashes; its own report documents
the later metadata-only correction and records before/after content hashes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from station_metadata import PROVENANCE, station_distance_km, station_metadata_records

BASE = Path(os.environ.get('RAINFALL_PROJECT_DIR', Path(__file__).resolve().parent.parent))
OUT = Path(os.environ.get('RAINFALL_OUTPUT_DIR', BASE / 'outputs'))
DISTANCE_COLUMNS = {'dist_km', 'distance_km'}


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def parse_csv(blob: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(blob.decode('utf-8-sig'), newline='')))


def nondistance_cells(rows: list[list[str]]) -> list[list[str]]:
    omitted = {i for i, name in enumerate(rows[0]) if name in DISTANCE_COLUMNS}
    return [[value for i, value in enumerate(row) if i not in omitted] for row in rows]


def prepare_refresh(blob: bytes) -> tuple[bytes, dict] | None:
    """Return changed bytes and verification, or None when no distance exists."""
    rows = parse_csv(blob)
    if not rows or not DISTANCE_COLUMNS.intersection(rows[0]):
        return None
    header = rows[0]
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV header prevents a safe metadata-only update')
    if 'station' not in header:
        raise ValueError('Distance-bearing CSV has no authoritative station column')
    si = header.index('station')
    di = [i for i, col in enumerate(header) if col in DISTANCE_COLUMNS]
    revised = [header.copy()]
    changed = []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f'CSV record {line} has an unexpected column count')
        new = row.copy()
        distance = format(station_distance_km(row[si]), '.12f')
        for i in di:
            if new[i] != distance:
                changed.append({'record': line, 'station': row[si], 'column': header[i],
                                'before': row[i], 'after': distance})
                new[i] = distance
        revised.append(new)
    if nondistance_cells(rows) != nondistance_cells(revised):
        raise AssertionError('Non-distance data changed before serialization')
    if not changed:
        return blob, {'changed_cells': [], 'row_count': len(rows)-1,
                      'non_distance_cells_identical': True}
    line_ending = '\r\n' if b'\r\n' in blob else '\n'
    buffer = io.StringIO(newline='')
    csv.writer(buffer, lineterminator=line_ending).writerows(revised)
    result = buffer.getvalue().encode('utf-8')
    if blob.startswith(b'\xef\xbb\xbf'):
        result = b'\xef\xbb\xbf' + result
    check = parse_csv(result)
    if rows[0] != check[0] or len(rows) != len(check):
        raise AssertionError('CSV shape/header changed')
    if nondistance_cells(rows) != nondistance_cells(check):
        raise AssertionError('A non-distance cell changed during serialization')
    non_distance_hash = sha256(json.dumps(nondistance_cells(rows), ensure_ascii=False,
                                         separators=(',', ':')).encode('utf-8'))
    return result, {'changed_cells': changed, 'row_count': len(rows)-1,
                    'non_distance_cells_identical': True,
                    'non_distance_cells_sha256': non_distance_hash}


def _replace_atomic(path: Path, blob: bytes) -> None:
    # The temporary file is in the checked target directory for atomic replace.
    with tempfile.NamedTemporaryFile(prefix='.distance_', suffix='.tmp', dir=path.parent,
                                     delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply verified metadata-only corrections')
    parser.add_argument('--output-dir', type=Path, default=OUT, help='Specific existing run directory')
    args = parser.parse_args()
    root = args.output_dir.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError('OUT must be the specific run directory, not a drive root')
    plans = []
    files_scanned = 0
    # The allowlist scopes traversal to numerical output CSVs and their supplement
    # copies, excluding any metadata backups created by this utility on reruns.
    candidates = sorted(root.glob('*.csv'))
    supplement = root / 'supplementary_tables'
    if supplement.is_dir():
        candidates += sorted(supplement.glob('*.csv'))
    for path in candidates:
        if path.is_symlink():
            raise ValueError(f'Refusing CSV symbolic link: {path}')
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
        original = path.read_bytes()
        files_scanned += 1
        prepared = prepare_refresh(original)
        if prepared is None:
            continue
        revised, detail = prepared
        detail.update({'file': str(relative), 'sha256_before': sha256(original),
                       'sha256_after': sha256(revised), 'changed': original != revised})
        plans.append((resolved, original, revised, detail))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_dir = root / f'distance_metadata_backup_{stamp}'
    report = {'created_utc': stamp, 'mode': 'apply' if args.apply else 'dry_run',
              'output_directory': str(root), 'provenance': PROVENANCE,
              'station_metadata': station_metadata_records(), 'files_scanned': files_scanned,
              'scope': 'Only dist_km/distance_km cells; no model fit, label, feature, prediction or score changed.',
              'original_fit_provenance_retained': True,
              'files': [item[3] for item in plans],
              'all_non_distance_cells_identical': True}
    if args.apply:
        # Check every original again before backing up and replacing anything.
        for path, original, _, _ in plans:
            if path.read_bytes() != original:
                raise RuntimeError(f'File changed during preflight: {path}')
        for path, original, revised, detail in plans:
            if original == revised:
                continue
            target_backup = backup_dir / path.relative_to(root)
            target_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_backup)
            if target_backup.read_bytes() != original:
                raise AssertionError(f'Backup validation failed: {path}')
            _replace_atomic(path, revised)
            actual = path.read_bytes()
            if actual != revised or nondistance_cells(parse_csv(original)) != nondistance_cells(parse_csv(actual)):
                # Restore this one file from known original bytes; retain backup.
                _replace_atomic(path, original)
                raise AssertionError(f'Post-write check failed and file restored: {path}')
            detail['backup'] = str(target_backup.relative_to(root))
            detail['verified_after_write'] = True
    report_path = root / f'distance_metadata_refresh_{stamp}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report_path)
    print(json.dumps({'mode':report['mode'], 'distance_files':len(plans),
                      'changed_files':sum(p[1] != p[2] for p in plans),
                      'non_distance_cells_identical':True}))


if __name__ == '__main__':
    main()
