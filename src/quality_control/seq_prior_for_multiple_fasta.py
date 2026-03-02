import argparse
import logging
from pathlib import Path
from typing import Dict, List, Union

from quality_control.seq_prioritization import read_fasta, write_fasta, parse_score, FastaRec


def setup_logger(verbosity: int = 0) -> logging.Logger:
    """
    verbosity:
      0 -> INFO
      1 -> DEBUG
    """
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logger = logging.getLogger("seq_prioritization")
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def process_single_fasta(input_fasta: Union[str, Path], args: argparse.Namespace, logger: logging.Logger) -> None:
    input_fasta = Path(input_fasta)

    base_name = input_fasta.name
    name = input_fasta.stem  # safer than split(".")[0]
    raw = read_fasta(input_fasta)

    parsed: List[FastaRec] = []
    skipped_no_score = 0
    skipped_len = 0

    for h, s in raw:
        seqlen = len(s)
        if not (args.min_len <= seqlen <= args.max_len):
            skipped_len += 1
            continue

        sc = parse_score(h)
        if sc is None:
            skipped_no_score += 1
            continue

        parsed.append(FastaRec(header=h, seq=s, score=sc))

    if args.dedup:
        # keep best (lowest score) record per identical sequence
        best_by_seq: Dict[str, FastaRec] = {}
        for r in parsed:
            prev = best_by_seq.get(r.seq)
            if prev is None or r.score < prev.score:
                best_by_seq[r.seq] = r
        parsed = list(best_by_seq.values())

    parsed.sort(key=lambda r: r.score)
    top = parsed[: max(0, args.top_n)]

    # Rewrite headers to include rank + score, while preserving original header content
    out: List[FastaRec] = []
    for idx, r in enumerate(top, start=1):
        new_header = f"rank={idx} score={r.score:.6f} | {r.header}"
        out.append(FastaRec(header=new_header, seq=r.seq, score=r.score))

    out_path = args.out_fa_dir / f"{name}.fa"
    write_fasta(out_path, out)

    # Log summary
    logger.info("Processed %s", base_name)
    logger.info("Read records: %d", len(raw))
    logger.info("Kept (scored) records: %d", len(parsed))
    if args.dedup:
        logger.info("After dedup: %d", len(parsed))
    logger.info("Wrote top N: %d -> %s", len(out), out_path)

    if skipped_no_score:
        logger.warning("Skipped (no score in header): %d", skipped_no_score)
    if skipped_len:
        logger.warning("Skipped (length filter): %d", skipped_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_fa_dir", required=True, help="Input FASTA dir from ProteinMPNN")
    ap.add_argument("-o", "--out_fa_dir", required=True, help="Output FASTA dir for top sequences")
    ap.add_argument("-n", "--top_n", type=int, default=50, help="Number of top sequences to keep")
    ap.add_argument("--dedup", action="store_true", help="Remove exact duplicate sequences (keep best score)")
    ap.add_argument("--min_len", type=int, default=1, help="Minimum sequence length to keep")
    ap.add_argument("--max_len", type=int, default=10**9, help="Maximum sequence length to keep")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)")
    args = ap.parse_args()

    logger = setup_logger(args.verbose)

    in_path = Path(args.in_fa_dir)
    args.out_fa_dir = Path(args.out_fa_dir)
    args.out_fa_dir.mkdir(parents=True, exist_ok=True)

    # Collect input FASTA files
    exts = ("*.fa", "*.fasta", "*.faa")
    fasta_paths_list = [f for ext in exts for f in in_path.glob(ext)]

    if not fasta_paths_list:
        logger.warning("No FASTA files found in %s matching %s", in_path, ", ".join(exts))
        return

    logger.info("Found %d FASTA files in %s", len(fasta_paths_list), in_path)

    for fasta_path in fasta_paths_list:
        process_single_fasta(fasta_path, args, logger)


if __name__ == "__main__":
    main()
