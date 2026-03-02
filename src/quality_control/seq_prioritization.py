#!/usr/bin/env python3
"""
Rank ProteinMPNN-sampled sequences by `score` in FASTA headers and write top N to a new FASTA.

Works with headers like:
>T=0.1, sample=2, score=0.7403, global_score=1.5856, seq_recovery=0.5000
SEQUENCE

It will:
- parse `score=...` (float) from each header
- sort ascending (lower is better)
- keep top N
- write to output FASTA (adds rank + score to headers)

Usage:
  python rank_mpnn_fasta.py -i input.fasta -o top.fasta -n 50
Optional:
  --dedup           remove exact duplicate sequences (keeps best score among duplicates)
  --min_len 8       drop sequences shorter than this
  --max_len 200     drop sequences longer than this
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


SCORE_RE = re.compile(r"(?:^|[\s,])score\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


@dataclass
class FastaRec:
    header: str
    seq: str
    score: float


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: List[str] = []

    def flush():
        nonlocal header, seq_lines
        if header is not None:
            seq = "".join(seq_lines).replace(" ", "").replace("\t", "").upper()
            records.append((header, seq))
        header = None
        seq_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                seq_lines.append(line.strip())
    flush()
    return records


def parse_score(header: str) -> Optional[float]:
    m = SCORE_RE.search(header)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def write_fasta(path: str, recs: List[FastaRec], line_width: int = 80) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(f">{r.header}\n")
            seq = r.seq
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


