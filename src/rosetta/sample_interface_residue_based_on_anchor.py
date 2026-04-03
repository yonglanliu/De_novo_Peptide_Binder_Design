import argparse
import glob
import math
import os
import sys
import csv
import itertools
from itertools import chain
import numpy as np
import rmsd
import re
from multiprocessing import freeze_support

import pyrosetta
from pyrosetta import Pose, get_score_function
from pyrosetta import rosetta
from pyrosetta.rosetta import core, protocols, numeric

from pyrosetta.rosetta.core.chemical import (
    UPPER_TERMINUS_VARIANT,
    LOWER_TERMINUS_VARIANT,
    CUTPOINT_LOWER,
    CUTPOINT_UPPER,
)
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue
from pyrosetta.rosetta.protocols.protein_interface_design.filters import HbondsToResidueFilter
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector,
    ChainSelector,
    AndResidueSelector,
    NeighborhoodResidueSelector,
    ResiduePDBInfoHasLabelSelector,
)
from pyrosetta.rosetta.protocols.hbnet import UnsatSelector
from pyrosetta.rosetta.protocols.simple_filters import ShapeComplementarityFilter
from pyrosetta.rosetta.protocols.simple_ddg import DdgFilter
from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.core.pack.rotamer_set import bb_independent_rotamers
from pyrosetta.rosetta.core.conformation import Residue


# global variable to turn on debug mode
_DEBUG = True

# Defining global lists
L_res = [
    "GLY", "ALA", "CYS", "MET", "VAL", "LEU", "ILE", "ASP", "GLU", "ASN",
    "GLN", "THR", "SER", "TYR", "TRP", "PHE", "LYS", "ARG", "HIS", "PRO"
]
D_res = [
    "GLY", "DALA", "DCYS", "DMET", "DVAL", "DLEU", "DILE", "DASP", "DGLU",
    "DASN", "DGLN", "DTHR", "DSER", "DTYR", "DTRP", "DPHE", "DLYS", "DARG",
    "DHIS", "DPRO"
]


def rrange(n):
    """Rosetta-style 1-based range."""
    return range(1, n + 1)


def variant_remove(p):
    """Remove variant types from pose."""
    for ir in rrange(p.size()):
        if p.residue(ir).has_variant_type(UPPER_TERMINUS_VARIANT):
            remove_variant_type_from_pose_residue(p, UPPER_TERMINUS_VARIANT, ir)
        if p.residue(ir).has_variant_type(LOWER_TERMINUS_VARIANT):
            remove_variant_type_from_pose_residue(p, LOWER_TERMINUS_VARIANT, ir)
        if p.residue(ir).has_variant_type(CUTPOINT_LOWER):
            remove_variant_type_from_pose_residue(p, CUTPOINT_LOWER, ir)
        if p.residue(ir).has_variant_type(CUTPOINT_UPPER):
            remove_variant_type_from_pose_residue(p, CUTPOINT_UPPER, ir)


def extend(n, p, a, c_, n_, args):
    """Extend anchor residue. Assumes one chain.

    Args:
        n: number to add to pose
        p: pose
        a: residue to add
        c_: boolean, whether to add to c_terminal
        n_: boolean, whether to add to n_terminal
    """
    variant_remove(p)

    if n_:
        for _ in range(n):
            p.prepend_polymer_residue_before_seqpos(a, 1, True)
        if _DEBUG:
            p.dump_pdb(f"{args.outdir}/prepend_test.pdb")
            for res_no in range(1, n + 1):
                p.set_omega(res_no, 180.0)

    if c_:
        for _ in range(n):
            p.append_residue_by_bond(a, True)
            if _DEBUG:
                p.dump_pdb(f"{args.outdir}/append_test.pdb")
        for res_no in range(1, p.size() + 1):
            p.set_omega(res_no, 180.0)

    if _DEBUG:
        p.dump_pdb(f"{args.outdir}/extended.pdb")


def bin_sample(p, res, phi, psi, args):
    """Sample phi and psi torsion for residue res in pose p."""
    p.set_phi(res, phi)
    p.set_psi(res, psi)

    if _DEBUG:
        p.dump_pdb(f"{args.outdir}/set_{phi}_{psi}.pdb")


def relax(pose, c, args):
    """Relax chain c of pose using cartesian minimization."""
    indeces = []
    for resNo in rrange(pose.size()):
        if pose.residue(resNo).chain() == c:
            indeces.append(resNo)

    if not indeces:
        return

    my_score = get_score_function()

    # set cart weights
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_ring, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_proper, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_improper, 1.0)

    # set metal constraints
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.metalbinding_constraint, 1.0)

    frm = pyrosetta.rosetta.protocols.relax.FastRelax(my_score)

    # Keep constraint strengths constant throughout the relaxation
    frm.ramp_down_constraints(False)
    frm.cartesian(True)

    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_jump(1, False)
    mm.set_chi_true_range(indeces[0], indeces[-1])
    frm.set_movemap(mm)
    frm.apply(pose)

    pyrosetta.rosetta.core.pose.remove_lower_terminus_type_from_pose_residue(pose, 1)


def unsat_count(pose, chain):
    """Very rough function that checks for unsatisfied hbond donors or acceptors at interface."""
    indeces = []
    for resNo in rrange(pose.size()):
        if pose.residue(resNo).chain() == chain:
            indeces.append(resNo)

    if not indeces:
        return 0

    to_check = ",".join(map(str, indeces))
    index_sel = ResidueIndexSelector(to_check)
    unsats = UnsatSelector()
    unsats.set_consider_mainchain_only(False)
    unsats.set_scorefxn(get_score_function())
    unsat_chain = AndResidueSelector(index_sel, unsats)
    subset = unsat_chain.apply(pose)

    all_unsats = []
    for i in rrange(pose.size()):
        nearby = []
        if subset[i]:
            r1 = pyrosetta.rosetta.core.conformation.Residue(pose.residue(i))
            for j in rrange(pose.size()):
                if abs(j - i) > 1:
                    for at in range(1, pose.residue(j).natoms()):
                        d_sq = r1.xyz("O").distance_squared(pose.xyz(core.id.AtomID(at, j)))
                        if d_sq < 25.01:
                            nearby.append([i, j])
        if len(nearby) > 30:
            all_unsats.append(i)

    return len(all_unsats)


def metrics(p, c):
    """Score the pose and output different interface metrics."""
    scf = get_score_function()
    scf(p)

    tot_energy_value = p.energies().total_energies()[core.scoring.ScoreType.total_score]

    pep_energy_value = 0.0
    for resNo in rrange(p.size()):
        if p.residue(resNo).chain() == c:
            pep_energy_value += p.energies().residue_total_energies(
                p.residue(resNo).seqpos()
            )[core.scoring.ScoreType.total_score]

    sc = ShapeComplementarityFilter()
    sc.jump_id(1)
    sc_value = sc.report_sm(p)

    ddg1 = DdgFilter(0, get_score_function(), 1)
    ddg1.repack(0)
    ddg_value1 = ddg1.report_sm(p)

    ddg2 = DdgFilter(0, get_score_function(), 1)
    ddg2.repack(1)
    ddg_value2 = ddg2.report_sm(p)

    unsat_value = unsat_count(p, c)

    metrics_out = [
        ("total energy", tot_energy_value),
        ("peptide energy", pep_energy_value),
        ("shape complementarity", sc_value),
        ("ddg no_repack", ddg_value1),
        ("ddg repack", ddg_value2),
        ("number unsats", unsat_value),
    ]
    return metrics_out

def write_score_row(csv_path, metadata, metrics_out):
    row = list(metadata) + [value for _, value in metrics_out]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(row)

def coord_find(p, ir, ia):
    """Find coordinate of atom ia in residue ir of pose p."""
    name = re.compile(".HA ")
    if len(re.findall(name, ia)) == 0:
        coord_xyz = p.xyz(core.id.AtomID(p.residue(ir).atom_index(ia), ir))
        coord_arr = [coord_xyz[0], coord_xyz[1], coord_xyz[2]]

        if _DEBUG:
            print(ia, coord_arr)

        return coord_arr

def find_cent(A):
    """Find the center of mass of coordinates A."""
    sumA = [0, 0, 0]
    for i in range(len(A)):
        sumA[0] = sumA[0] + A[i][0]
        sumA[1] = sumA[1] + A[i][1]
        sumA[2] = sumA[2] + A[i][2]

    for i in range(3):
        sumA[i] = sumA[i] / len(A)

    if _DEBUG:
        print("========= find center function==========")
        print("the elements are", "\n", A, "\n", "and the center is:", "\n", sumA)

    return sumA


def transform(target_pose, rotate_pose, resn_in_target, resn_in_rotate):
    """
    Align residue resn_in_rotate in rotate_pose onto residue resn_in_target in target_pose using heavy side-chain atoms.
    
    target_pose: pose align to
    rotate_pose: pose align from
    resn_in_target: rosetta residue number in target_pose
    resn_in_rotate: rosetta residue number in rotate_pose
    """
    def heavy_sidechain_atom_names(pose, resn):
        res = pose.residue(resn)
        names = []
        for atom in range(1, res.natoms() + 1):
            if not res.atom_is_backbone(atom) and not res.atom_is_hydrogen(atom):
                names.append(res.atom_name(atom).strip())
        return names

    # Collect matching atom coordinates by atom name
    atom_names = heavy_sidechain_atom_names(target_pose, resn_in_target)

    c_target = []
    c_rotate = []
    for name in atom_names:
        # assumes the atom exists in both residues
        c_target.append(coord_find(target_pose, resn_in_target, name))
        c_rotate.append(coord_find(rotate_pose, resn_in_rotate, name))

    # Centroids
    c_t = find_cent(c_target)
    c_r = find_cent(c_rotate)

    # Center coordinates
    c_target_centered = [np.array(v) - np.array(c_t) for v in c_target]
    c_rotate_centered = [np.array(v) - np.array(c_r) for v in c_rotate]

    # Rotation from Kabsch
    R = np.linalg.inv(rmsd.kabsch(c_rotate_centered, c_target_centered))

    # Apply: x' = R * (x - c_r) + c_t
    Rx = numeric.xyzMatrix_double_t.cols(
        R[0][0], R[1][0], R[2][0],
        R[0][1], R[1][1], R[2][1],
        R[0][2], R[1][2], R[2][2],
    )

    v1 = numeric.xyzVector_double_t(-c_r[0], -c_r[1], -c_r[2])
    v2 = numeric.xyzVector_double_t(c_t[0], c_t[1], c_t[2])

    # Create matrix without rotation
    noR = numeric.xyzMatrix_double_t.cols(1, 0, 0, 
                                          0, 1, 0, 
                                          0, 0, 1)

    rotate_pose.apply_transform_Rx_plus_v(noR, v1) # Translocated to the center without rotation
    rotate_pose.apply_transform_Rx_plus_v(Rx, numeric.xyzVector_double_t(0, 0, 0)) # Rotate aling Rx without tranlocation
    rotate_pose.apply_transform_Rx_plus_v(noR, v2) # Transloated to target center without rotation

    if _DEBUG:
        aligned = [
            coord_find(rotate_pose, resn_in_rotate, name)
            for name in atom_names
        ]
        print("final rmsd:", rmsd.kabsch_rmsd(aligned, c_target))


def packer(pose, res_name, res_num, chain_id):
    """
    Perform mutation and minimization.
    pose: input rosetta pose
    res_name: residue name mutate to
    res_num: rosetta residue number
    chain_id: peptide chain id
    """
    indice = []
    for resNo in rrange(pose.size()):
        if pose.residue(resNo).chain() == chain_id:
            indice.append(resNo)

    # step one is to add the residue you want
    mut = protocols.simple_moves.MutateResidue()
    mut.set_res_name(res_name)
    mut.set_target(res_num)
    mut.set_preserve_atom_coords(True)
    mut.apply(pose)

    # step two is to minimize with a cart_min probably
    my_score = get_score_function()

    # set cart weights
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_ring, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_proper, 1.0)
    my_score.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_improper, 1.0)

    min_mover = protocols.minimization_packing.MinMover()
    min_mover.score_function(my_score)
    min_mover.cartesian(True)

    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_jump(1, False) # do not move jump 1
    mm.set_bb_true_range(indice[0], indice[-1]) # Enables backbone movement (ϕ/ψ angles)
    mm.set_chi_true_range(indice[0], indice[-1]) # Enables side-chain movement (χ angles)
    min_mover.set_movemap(mm)
    min_mover.apply(pose)


##############################################################################################
########################## MAIN FUNCTION THAT CALLS EVERYTHING ELSE ##########################
##############################################################################################
def main(argv):
    parser = argparse.ArgumentParser(description="Program")
    parser.add_argument(
        "-i", "--input", action="store", type=str,
        required=True,
        help="input target pdb",
    )
    parser.add_argument(
        "-o", "--outdir", action="store", type=str,
        required=True,
        help="directory for storing output files",
    )
    parser.add_argument(
        "-n", "--number", action="store", type=int,
        required=True,
        help="number of extension at each end",
    )
    parser.add_argument(
        "-m", "--mut", action="store", type=int,
        required=True,
        help="residue index to run all the samples on",
    )
    parser.add_argument(
        "-a", "--aa", action="store", type=str,
        default="GLY",
        help="aa type to append",
    )
    parser.add_argument(
        "-c", "--chain", action="store", type=int,
        default=1,
        help="what chain number is the one I am extending",
    )
    parser.add_argument(
        "-ep", "--extra_parms",
        type=lambda s: s.split(","),
        default=[],
        help="extra params files",
    )
    parser.add_argument(
        "-ar", "--anchors",
        type=lambda s: list(map(int, s.split(","))),
        default=[2],
        help="anchors as comma-separated list (e.g. -ar 2,3)",
    )
    parser.add_argument(
        "-r", "--relax", action="store_true",
        help="should I run relax",
    )
    

    args = parser.parse_args()

    # list of additional param files one wants to add.
    params = []

    # user-provided params
    for f in args.extra_parms:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Params file not found: {f}")
        params.append(f)
    # auto-detect params in cwd (optional)
    params.extend(sorted(glob.glob("*.params")))

    # deduplicate while preserving order
    params = list(dict.fromkeys(params))

    # initiating Rosetta
    init_options = [
        "-beta_nov16_cart",
        "-in:file:fullatom true",
        "-write_all_connect_info",
        "-ignore_waters false",
        "-mute all",
        "-auto_setup_metals true",
    ]
    if params:
        init_options.append(f"-in:file:extra_res_fa {' '.join(params)}")

    pyrosetta.init(" ".join(init_options))
    scorefxn = get_score_function()

    # get the pose from target and scaffold
    p_in = rosetta.core.import_pose.pose_from_file(args.input)

    # ---defining the residue I want to append---
    chm = rosetta.core.chemical.ChemicalManager.get_instance()
    rts = chm.residue_type_set("fa_standard")
    res = rosetta.core.conformation.ResidueFactory.create_residue(
        rts.name_map(args.aa)
    )
    # --------------------------------------------

    # ---Create pose for peptide and extend residue from two sides of peptide---
    p = Pose()
    for resNo in rrange(p_in.size()):
        if p_in.residue(resNo).chain() == args.chain:
            p.append_residue_by_bond(p_in.residue(resNo), False) # do not automatically build ideal bond geometry

    if _DEBUG:
        p.dump_pdb(f"{args.outdir}/just_pep.pdb")

    extend(args.number, p, res, True, True, args) # Extend: number of residues to extend at the two side
                                            # p: peptide pose
                                            # res: extend residue name
                                            # Nterminal and Cterminal
    # --------------------------------------------

    # starting from anchor
    pep_s = Pose()
    for resNo in rrange(p.size()):
        pep_s.append_residue_by_bond(p.residue(resNo), False) # do not automatically build ideal bond geometry

    if _DEBUG:
        pep_s.dump_pdb(f"{args.outdir}/clone_check.pdb") # pep_s is peptide pose

    score_path = f"{args.outdir}/scores_{args.mut}.csv"

    with open(score_path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow([
            "residue_position",
            "aa",
            "phi",
            "psi",
            "sampled_phi",
            "sampled_psi",
            "total_score",
            "peptide_total_score",
            "sc",
            "ddg_no_repack",
            "ddg_repack",
            "unsats",
        ])
    anchor_res_num = args.anchors

    # going through all the residues, all changes slowly.
    for phi in range(-180, 181, 10):
        if phi > -30 and phi < 30:
            continue
        for psi in range(-180, 181, 10):
            bin_sample(p, args.mut, phi, psi, args)

            # after sampling making sure anchor residues are still there
            for n in anchor_res_num:
                transform(pep_s, p, n, n) 

            # adding things back as one chain
            variant_remove(p_in)
            p_fin = Pose()
             

            # Add peptide chain first
            for resi in rrange(p.size()):
                p_fin.append_residue_by_bond(p.residue(resi), False)

            # Add the rest of pose back chain-by-chain
            prev_chains = set([args.chain])

            for resNo in rrange(p_in.size()):
                if p_in.residue(resNo).chain() == args.chain: # skip peptide chain 1
                    continue
                if p_in.residue(resNo).chain() not in prev_chains: 
                    prev_chains.add(p_in.residue(resNo).chain())
                    p_fin.append_residue_by_jump(p_in.residue(resNo), p_fin.size(),"", "", True) # True: start a new chain
                else:
                    p_fin.append_residue_by_bond(p_in.residue(resNo), False) # False: Not build ideal geometry--preserve existing coordinates

            if _DEBUG:
                p_fin.dump_pdb(f"{args.outdir}/moved_{phi}_{psi}.pdb")

            if phi > 0:
                for resNa in D_res:
                    packer(p_fin, resNa, args.mut, args.chain)
                    write_score_row(score_path,[args.mut, resNa, phi, psi, p.phi(args.mut), p.psi(args.mut)], metrics(p_fin, 1))
            else:
                for resNa in L_res:
                    packer(p_fin, resNa, args.mut, args.chain)
                    write_score_row(score_path,[args.mut, resNa, phi, psi, p.phi(args.mut), p.psi(args.mut)], metrics(p_fin, 1))

    if args.relax:
        relax(p_fin, args.chain)

    if _DEBUG:
        p_fin.dump_pdb(f"{args.outdir}/final.pdb")


if __name__ == "__main__":
    freeze_support()
    main(sys.argv)
