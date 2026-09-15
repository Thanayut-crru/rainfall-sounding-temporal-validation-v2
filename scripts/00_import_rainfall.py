# -*- coding: utf-8 -*-
"""Convert TMD 3-hour rainfall workbooks to the daily input used by the paper.

The parser uses only the Python standard library so it can read the original
XLSX file without requiring Excel or openpyxl. Eight local-time observations
(01, 04, 07, 10, 13, 16, 19, and 22 h) are summed for each calendar day.

Conventions required by the manuscript:
* ``T`` (trace) is treated as 0.0 mm; it remains below the 0.1-mm rain-day
  threshold.
* ``-`` or an absent 3-hour observation makes that daily total missing.
* A dash in the workbook's daily-total column is accepted only as a display
  convention for a complete all-zero day; totals are recomputed from the eight
  component observations rather than copied from that column.

Station mapping:
* 564201 -> Phuket
* 566202 -> Krabi
* 561201 -> Takua Pa (Phang-nga Province), reported as Phang-nga in the paper
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
M = f"{{{MAIN_NS}}}"

STATIONS = {
    "564201": ("phuket", "RF_phuket"),
    "566202": ("krabi", "RF_krabi"),
    "561201": ("phangnga", "RF_phangnga"),
}
EXPECTED_HOURS = ["100", "400", "700", "1000", "1300", "1600", "1900", "2200"]
INTERVAL_COLUMNS = list("DEFGHIJK")


def cell_column(reference: str) -> str:
    return re.match(r"[A-Z]+", reference).group(0)


def excel_date(value: str):
    return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()


def parse_amount(value: str | None) -> tuple[float | None, str]:
    if value is None or value.strip() == "-":
        return None, "missing"
    if value.strip().upper() == "T":
        return 0.0, "trace"
    try:
        amount = float(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported rainfall token: {value!r}") from exc
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"Invalid rainfall amount: {value!r}")
    return amount, "numeric"


def shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter(M + "t"))
            for item in root.findall(M + "si")]


def worksheet_paths(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {node.attrib["Id"]: node.attrib["Target"] for node in rels}
    paths = []
    for sheet in workbook.find(M + "sheets"):
        rid = sheet.attrib[f"{{{REL_NS}}}id"]
        target = targets[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        paths.append((sheet.attrib["name"], target))
    return paths


def row_values(row: ET.Element, strings: list[str]) -> dict[str, str | None]:
    values = {}
    for cell in row.findall(M + "c"):
        column = cell_column(cell.attrib["r"])
        raw = cell.find(M + "v")
        value = None if raw is None else raw.text
        if cell.attrib.get("t") == "s" and value is not None:
            value = strings[int(value)]
        elif cell.attrib.get("t") == "inlineStr":
            inline = cell.find(M + "is")
            value = "" if inline is None else "".join(
                node.text or "" for node in inline.iter(M + "t"))
        values[column] = value
    return values


def find_three_hour_sheet(archive: ZipFile, strings: list[str]) -> tuple[str, ET.Element]:
    for name, path in worksheet_paths(archive):
        root = ET.fromstring(archive.read(path))
        rows = root.findall(".//" + M + "sheetData/" + M + "row")
        for row in rows[:10]:
            if row.attrib.get("r") == "5":
                values = row_values(row, strings)
                if [values.get(c) for c in INTERVAL_COLUMNS] == EXPECTED_HOURS:
                    return name, root
    raise ValueError("Could not locate the worksheet with eight 3-hour observations")


def read_workbook(path: Path) -> tuple[list[dict], dict]:
    records = []
    with ZipFile(path) as archive:
        strings = shared_strings(archive)
        sheet_name, root = find_three_hour_sheet(archive, strings)
        rows = root.findall(".//" + M + "sheetData/" + M + "row")
        for row in rows:
            if int(row.attrib.get("r", "0")) <= 5:
                continue
            values = row_values(row, strings)
            station_text = values.get("B") or ""
            station_code = station_text.split("-", 1)[0].strip()
            if station_code not in STATIONS or not values.get("C"):
                continue
            amounts, states = zip(*(parse_amount(values.get(c)) for c in INTERVAL_COLUMNS))
            complete = all(amount is not None for amount in amounts)
            daily_total = round(sum(amounts), 3) if complete else None
            records.append({
                "date": excel_date(values["C"]),
                "station_code": station_code,
                "daily_total": daily_total,
                "complete": complete,
                "n_trace": states.count("trace"),
                "n_missing_intervals": states.count("missing"),
                "interval_amounts": dict(zip(EXPECTED_HOURS, amounts)),
            })
    return records, {"file": str(path), "sheet": sheet_name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="TMD XLSX file; repeat this option to combine years")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "rainfall_tmd.csv")
    parser.add_argument("--qc-output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "rainfall_import_qc.csv")
    parser.add_argument("--metadata-output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "rainfall_import_metadata.json")
    parser.add_argument("--interval-output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "rainfall_tmd_3hour.csv")
    args = parser.parse_args()

    records, sources = [], []
    for input_path in args.input:
        rows, source = read_workbook(input_path.resolve())
        records.extend(rows)
        sources.append(source)

    keyed = {}
    identical_duplicates_removed = 0
    for row in records:
        key = (row["date"], row["station_code"])
        if key in keyed:
            comparable = ("daily_total", "complete", "n_trace", "n_missing_intervals", "interval_amounts")
            if all(keyed[key][field] == row[field] for field in comparable):
                identical_duplicates_removed += 1
                continue
            raise ValueError(f"Conflicting duplicate station-date: {key}")
        keyed[key] = row
    records = list(keyed.values())

    dates = sorted({row["date"] for row in records})
    wide = defaultdict(dict)
    for row in records:
        wide[row["date"]][row["station_code"]] = row["daily_total"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "RF_phuket", "RF_krabi", "RF_phangnga"])
        writer.writeheader()
        for date in dates:
            writer.writerow({
                "date": date.isoformat(),
                **{column: wide[date].get(code) for code, (_, column) in STATIONS.items()},
            })

    # Preserve the eight original local-time intervals for reviewer-requested
    # time-of-day verification. Missing intervals remain blank.
    with args.interval_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "station", "station_code", "local_hour", "rain_mm"])
        writer.writeheader()
        for row in sorted(records, key=lambda x: (x["date"], x["station_code"])):
            station = STATIONS[row["station_code"]][0]
            for hour in EXPECTED_HOURS:
                writer.writerow({"date": row["date"].isoformat(), "station": station,
                                 "station_code": row["station_code"], "local_hour": int(hour)//100,
                                 "rain_mm": row["interval_amounts"][hour]})

    qc_rows = []
    for code, (station, _) in STATIONS.items():
        subset = [row for row in records if row["station_code"] == code]
        years = Counter(row["date"].year for row in subset)
        for year in sorted(years):
            annual = [row for row in subset if row["date"].year == year]
            complete = [row for row in annual if row["complete"]]
            qc_rows.append({
                "station": station,
                "station_code": code,
                "year": year,
                "n_rows": len(annual),
                "n_complete_days": len(complete),
                "n_incomplete_days": len(annual) - len(complete),
                "n_rain_days_ge_0_1mm": sum(row["daily_total"] >= 0.1 for row in complete),
                "n_trace_observations": sum(row["n_trace"] for row in annual),
                "n_missing_3h_observations": sum(row["n_missing_intervals"] for row in annual),
            })
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    with args.qc_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)

    metadata = {
        "sources": sources,
        "station_mapping": {code: {"station": station, "column": column}
                            for code, (station, column) in STATIONS.items()},
        "interval_hours_local": EXPECTED_HOURS,
        "trace_value_mm": 0.0,
        "rain_day_threshold_mm": 0.1,
        "incomplete_day_policy": "NA if any of eight 3-hour observations is missing",
        "date_min": min(dates).isoformat(),
        "date_max": max(dates).isoformat(),
        "n_calendar_dates": len(dates),
        "identical_duplicate_records_removed": identical_duplicates_removed,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Saved {args.qc_output}")
    print(f"Saved {args.metadata_output}")
    print(f"Saved {args.interval_output}")


if __name__ == "__main__":
    main()
