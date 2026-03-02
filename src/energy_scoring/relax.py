#!/usr/bin/env python3

import csv
import logging
from pathlib import Path
import argparse
import os
import numpy as np
import pyrosetta
from typing import Iterable
from pyrosetta import rosetta, init, Pose, pose_from_pdb, pose_from_file

#################################
#      Initialization           #
#################################

def init_pyrosetta(extra_flags: str = "") -> None:
    """
    -mute all : supresses rosetta's verbose output
    -ex1 ex2 : enable extra rotamers -- extra chi1 ane extra chi2 samplings
    -use_input_sc : Use input side-chain conformations as starting rotamers. 
                    Without this, rosseta may ignore input PDB's side chains and rebuld them
    -relax:fast : Uses the "fast" relax protocol (fewer cycles, quicker runtime)
    """
    flags = (
        "-mute all "
        "-ex1 -ex2 "
        "-use_input_sc "
        "-relax:fast "
        f"{extra_flags} "
    )
    init(flags)


#################################
#          Set up loggers       #
#################################

def setup_logger(out_dir: Path, verbose: bool = False) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("binder_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(out_dir / "run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


#################################
#        Arg parsing utils      #
#################################

from typing import Iterable
import argparse


def parse_chains(chains: str | Iterable[str]) -> list[str]:
    """
    Normalize chain specification into a unique ordered list of chain IDs.

    Accepts:
        "A,C"      -> ["A","C"]
        "AC"       -> ["A","C"]
        "A C"      -> ["A","C"]
        ["A","C"]  -> ["A","C"]
        ["AC"]     -> ["A","C"]

    Raises:
        argparse.ArgumentTypeError if empty or invalid.
    """

    if chains is None:
        raise argparse.ArgumentTypeError("Chains input is None.")
    
    # If input is already iterable (list/tuple)
    if isinstance(chains, (list, tuple, set)):
        tokens = []
        for item in chains:
            if not isinstance(item, str):
                raise argparse.ArgumentTypeError(
                    f"Invalid chain entry: {item} (must be string)"
                )
            tokens.append(item.strip())

    # If input is string

    elif isinstance(chains, str):
        s = chains.strip()
        if not s:
            raise argparse.ArgumentTypeError("Chains string is empty.")

        # normalize common delimiters
        for delim in [";", " ", "|"]:
            s = s.replace(delim, ",")

        tokens = [p.strip() for p in s.split(",") if p.strip()]
    else:
        raise argparse.ArgumentTypeError(
            f"Unsupported type for chains: {type(chains)}"
        )

    # Expand tokens like "AC" → ["A","C"]
    expanded = []
    for token in tokens:
        expanded.extend(list(token))

    # Remove duplicates (preserve order)

    seen = set()
    unique = []
    for c in expanded:
        if c and c not in seen:
            unique.append(c)
            seen.add(c)

    if not unique:
        raise argparse.ArgumentTypeError("No valid chains parsed.")

    return unique


def make_interface_str(target_chains: list[str], binder_chains: list[str]) -> str:
    # common format: "A,C_B" (targets on left, binders on right)
    left = ",".join(target_chains)
    right = ",".join(binder_chains)
    return f"{left}_{right}"


#################################
#        Chain Selection        #
#################################

def chain_selector(chain: str):
    return rosetta.core.select.residue_selector.ChainSelector(chain)


#################################
#      CA RMSD for Chains       #
#################################

def chain_rmsd(pose: Pose, chains: list[str] | str, ref_pose: Pose) -> float:
    """
    Accepts:
      - "A,C" -> ["A","C"]
      - "AC"  -> ["A","C"]
      - "A C" -> ["A","C"]
    """
    chains = parse_chains(chains)
    ids = rosetta.utility.vector1_unsigned_long()  # Creates a Rosetta C++ container inside Python
    for c in chains:
        sel_chain = chain_selector(c)
        subset = sel_chain.apply(pose)  # 1-indexed vector1_bool
        for i in range(1, pose.total_residue() + 1):
            if subset[i]:
                ids.append(i)

    if len(ids) == 0:
        raise ValueError(f"No residues selected for chains {chains}. Check chain IDs in PDB.")
    return rosetta.core.scoring.CA_rmsd(pose, ref_pose, ids)


#################################
#      Interaction Score        #
#################################

def interface_energy(pose: Pose, scorefxn, interface_str: str) -> float:
    iam = rosetta.protocols.analysis.InterfaceAnalyzerMover(
        interface_str,
        False,      # do not compute packstat unless you want it
        scorefxn
    )
    iam.apply(pose)
    return float(iam.get_interface_dG())


#################################
#         Binder Energy         #
#################################

def binder_energy(pose: Pose, scorefxn, binder_chains: list[str] | str) -> float:
    """
    Accepts:
      - "A,C" -> ["A","C"]
      - "AC"  -> ["A","C"]
      - "A C" -> ["A","C"]
    """
    binder_chains = parse_chains(binder_chains)
    # ensure energies are up-to-date
    scorefxn(pose)

    total = 0.0
    for c in binder_chains:
        selector = chain_selector(c)
        subset = selector.apply(pose)
        for i in range(1, pose.total_residue() + 1):
            if subset[i]:
                total += float(pose.energies().residue_total_energy(i))
    return total


#################################
#           Fast Relax          #
#################################

def fast_relax(pose: Pose, scorefxn, max_iter: int = 200) -> None:
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.max_iter(max_iter)
    relax.apply(pose)


#################################
#      Binder perturbation      #
#################################

def perturb_binder(pose: Pose, binder_chains: list[str] | str, sigma: float = 0.5) -> None:
    binder_chains = parse_chains(binder_chains)
    dx, dy, dz = np.random.normal(0.0, sigma, 3)

    # build a union selection of binder chains
    selected = [False] * (pose.total_residue() + 1)  # 1-indexed convenience
    for c in binder_chains:
        subset = chain_selector(c).apply(pose)
        for i in range(1, pose.total_residue() + 1):
            if subset[i]:
                selected[i] = True

    for i in range(1, pose.total_residue() + 1):
        if not selected[i]:
            continue
        res = pose.residue(i)
        for a in range(1, res.natoms() + 1):
            v = res.xyz(a)
            pose.set_xyz(
                rosetta.core.id.AtomID(a, i),
                rosetta.numeric.xyzVector_double_t(v.x + dx, v.y + dy, v.z + dz)
            )


#################################
#      Main Pipeline Running    #
#################################

def run_pipeline(args) -> None:
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, verbose=args.verbose)

    logger.info("Initializing PyRosetta...")
    init_pyrosetta()

    logger.info("Loading input PDB: %s", args.input_pdb)
    pose0 = rosetta.core.pose.Pose()
    rosetta.core.import_pose.pose_from_file(pose0, args.input_pdb)

    ref_pose = pose0.clone()
    scorefxn = pyrosetta.get_fa_scorefxn()

    interface_str = make_interface_str(args.target_chains, args.design_chains)
    logger.info("Target chains: %s", ",".join(args.target_chains))
    logger.info("Binder chains: %s", ",".join(args.design_chains))
    logger.info("Interface string: %s", interface_str)

    scores_path = out_dir / "scores.csv"
    logger.info("Writing scores to: %s", str(scores_path))

    stem = Path(args.input_pdb).stem

    with open(scores_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "total_score", "interface_dG", "binder_score", "target_rmsd"])

        for k in range(args.n_out):
            pose = pose0.clone()

            perturb_binder(pose, binder_chains=args.design_chains, sigma=args.perturb_sigma)
            fast_relax(pose, scorefxn=scorefxn, max_iter=args.relax_iter)

            total_score = float(scorefxn(pose))
            dg = interface_energy(pose, scorefxn, interface_str=interface_str)
            binderE = binder_energy(pose, scorefxn, binder_chains=args.design_chains)
            target_r = chain_rmsd(pose, chains=args.target_chains, ref_pose=ref_pose)

            name = f"{stem}_{k:03d}.pdb"
            out_pdb = out_dir / name
            pose.dump_pdb(str(out_pdb))

            writer.writerow([name, total_score, dg, binderE, target_r])

            logger.info(
                "[%d/%d] %s | total=%.3f | dG=%.3f | binder=%.3f | rmsd=%.3f",
                k + 1, args.n_out, name, total_score, dg, binderE, target_r
            )

    logger.info("Done.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Perturb binder chains, relax, and score interface + RMSD.")
    p.add_argument("--input_pdb", type=str, required=True, help="Input PDB file")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--n_out", type=int, default=1, help="Number of output PDBs")
    p.add_argument("--design_chains", type=parse_chains, default=parse_chains("B"),
                   help='Binder chain IDs (e.g. "B" or "B,D")')
    p.add_argument("--target_chains", type=parse_chains, default=parse_chains("A"),
                   help='Target chain IDs (e.g. "A,C")')

    p.add_argument("--perturb_sigma", type=float, default=0.5, help="Translation sigma (Å) for binder perturbation")
    p.add_argument("--relax_iter", type=int, default=200, help="FastRelax max iterations")

    p.add_argument("--verbose", action="store_true", help="Verbose console logging")
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
