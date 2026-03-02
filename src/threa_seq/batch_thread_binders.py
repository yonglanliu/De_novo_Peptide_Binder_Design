#!/usr/bin/env python3
"""
Batch thread (assign) binder-chain sequences onto many complex PDBs using PyRosetta.

You have:
  - pdbdir: directory of complex PDBs (one complex per file)
  - seqdir: directory of multi-FASTA files, one per PDB (top-ranked sequences)
Matching rule (default):
  - PDB basename must match FASTA basename (ignoring extension)
    e.g. pdbdir/design_0001.pdb  <->  seqdir/design_0001.fasta

For each PDB + its FASTA:
  - For each sequence record in FASTA:
      - clone pose
      - thread sequence onto binder chain (default chain B)
      - write full-atom PDB (backbone unchanged; sidechains unrelaxed)

No packing/minimization is performed.

Usage:
  python batch_thread_binders.py \
    --pdbdir rf_pdbs \
    --seqdir top_fastas \
    --chain B \
    --outdir threaded_pdbs

Optional:
  --glob "*.pdb"
  --fasta_ext ".fa"     (default tries .fasta/.fa/.faa)
  --max_per_pdb 50
  --dedup              (dedup exact duplicate sequences within each FASTA; keeps first occurrence)
  --init "-beta_nov16 -mute all -use_terminal_residues true"
"""

import argparse
import os
import re
import glob
from typing import List, Tuple, Optional, Dict

from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta import core


AA_1_TO_3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}
VALID_AA = set(AA_1_TO_3.keys())


def read_multifasta(path: str) -> List[Tuple[str, str]]:
    recs: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: List[str] = []

    def flush():
        nonlocal header, seq_lines
        if header is not None:
            seq = "".join(seq_lines).replace(" ", "").replace("\t", "").upper()
            recs.append((header, seq))
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
                seq_lines.append(line)
    flush()
    return recs


def sanitize_name(s: str, maxlen: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return (s[:maxlen] if len(s) > maxlen else s) or "seq"


def pose_positions_for_chain(pose, chain_id: str) -> List[int]:
    pdbinfo = pose.pdb_info()
    if pdbinfo is None:
        raise RuntimeError("Pose has no PDBInfo; cannot map chain IDs.")
    idxs = [i for i in range(1, pose.total_residue() + 1) if pdbinfo.chain(i) == chain_id]
    if not idxs:
        raise ValueError(f"No residues found for chain '{chain_id}'. Check chain IDs in the PDB.")
    return idxs


def thread_sequence_onto_chain(pose, chain_positions: List[int], seq: str) -> None:
    if len(seq) != len(chain_positions):
        raise ValueError(f"Sequence length {len(seq)} != chain length {len(chain_positions)}")
    bad = sorted({aa for aa in seq if aa not in VALID_AA})
    if bad:
        raise ValueError(f"Unsupported AA letters: {bad} (only standard 20 AAs allowed)")

    rsd_set = pose.residue_type_set_for_pose(core.chemical.FULL_ATOM_t)
    for aa, pose_idx in zip(seq, chain_positions):
        name3 = AA_1_TO_3[aa]
        new_res = core.conformation.ResidueFactory.create_residue(rsd_set.name_map(name3))
        pose.replace_residue(pose_idx, new_res, True)  # keep bb, idealize sc


def find_matching_fasta(seqdir: str, base: str, forced_ext: Optional[str] = None) -> Optional[str]:
    if forced_ext:
        cand = os.path.join(seqdir, base + forced_ext)
        return cand if os.path.isfile(cand) else None

    for ext in [".fasta", ".fa", ".faa", ".fas"]:
        cand = os.path.join(seqdir, base + ext)
        if os.path.isfile(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbdir", required=True, help="Directory containing complex PDBs")
    ap.add_argument("--seqdir", required=True, help="Directory containing multi-FASTA files (one per PDB)")
    ap.add_argument("--outdir", default="threaded_outputs", help="Output directory")
    ap.add_argument("--chain", default="B", help="Binder chain ID in the PDB (default: B)")
    ap.add_argument("--glob", default="*.pdb", help="Glob for PDB files inside pdbdir (default: *.pdb)")
    ap.add_argument("--fasta_ext", default="", help="Force FASTA extension, e.g. .fasta or .fa (default: auto-detect)")
    ap.add_argument("--max_per_pdb", type=int, default=0,
                    help="Max sequences to thread per PDB (0 = all in FASTA)")
    ap.add_argument("--dedup", action="store_true",
                    help="Deduplicate exact duplicate sequences within each FASTA (keeps first)")
    ap.add_argument("--init", default="-beta_nov16 -mute all -use_terminal_residues true",
                    help="Extra PyRosetta init flags")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    init(args.init)

    pdb_paths = sorted(glob.glob(os.path.join(args.pdbdir, args.glob)))
    if not pdb_paths:
        raise RuntimeError(f"No PDBs found in {args.pdbdir} matching {args.glob}")

    forced_ext = args.fasta_ext if args.fasta_ext else None

    total_written = 0
    total_skipped = 0
    missing_fasta = 0

    print(f"Found {len(pdb_paths)} PDB(s)")

    for pdb_path in pdb_paths:
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        fasta_path = find_matching_fasta(args.seqdir, base, forced_ext=forced_ext)
        if fasta_path is None:
            missing_fasta += 1
            print(f"[MISS] No FASTA found for {base} in {args.seqdir}")
            continue

        try:
            base_pose = pose_from_pdb(pdb_path)
            chain_positions = pose_positions_for_chain(base_pose, args.chain)
        except Exception as e:
            total_skipped += 1
            print(f"[SKIP] Failed to load/map chain for {pdb_path}: {e}")
            continue

        try:
            records = read_multifasta(fasta_path)
        except Exception as e:
            total_skipped += 1
            print(f"[SKIP] Failed to read FASTA {fasta_path}: {e}")
            continue

        if args.dedup:
            seen = set()
            deduped = []
            for h, s in records:
                if s in seen:
                    continue
                seen.add(s)
                deduped.append((h, s))
            records = deduped

        if args.max_per_pdb and args.max_per_pdb > 0:
            records = records[: args.max_per_pdb]

        if not records:
            print(f"[SKIP] No sequences in FASTA for {base}: {fasta_path}")
            continue

        out_subdir = os.path.join(args.outdir, base)
        os.makedirs(out_subdir, exist_ok=True)

        chain_len = len(chain_positions)
        print(f"[OK] {base}: {len(records)} seq(s) | binder chain {args.chain} length {chain_len}")

        for rank, (hdr, seq) in enumerate(records, start=1):
            try:
                pose = base_pose.clone()
                thread_sequence_onto_chain(pose, chain_positions, seq)

                tag = sanitize_name(hdr)
                outname = f"{base}_chain{args.chain}_rank{rank}_{tag}.pdb"
                outpath = os.path.join(out_subdir, outname)

                pose.dump_pdb(outpath)
                total_written += 1

            except Exception as e:
                total_skipped += 1
                print(f"  [SEQ-SKIP] {base} rank {rank} ({hdr}): {e}")

    print("\n=== Summary ===")
    print(f"Wrote PDBs:      {total_written}")
    print(f"Skipped items:   {total_skipped}")
    print(f"Missing FASTAs:  {missing_fasta}")
    print(f"Output dir:      {args.outdir}")
    print("Note: outputs include sidechain atoms but are NOT packed/relaxed (may look clashing).")


if __name__ == "__main__":
    main()
