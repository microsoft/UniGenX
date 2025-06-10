# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
import random
import re
import statistics

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolops import RemoveHs
from tqdm import tqdm


def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd


def get_cov_mat(gen_list, ref_list, threshold=0.5, useFF=False):
    if gen_list == [] or ref_list == []:
        return None, None
    cov_count = 0
    mat_sum = 0
    for ref_mol in ref_list:
        rmsd_list = []
        for gen_mol in gen_list:
            if useFF is True:
                try:
                    MMFFOptimizeMolecule(gen_mol)
                except:
                    pass
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list) <= threshold:
            cov_count += 1
        mat_sum += min(rmsd_list)

    return 100 * cov_count / len(ref_list), mat_sum / len(ref_list)


def get_cov_mat_p(gen_list, ref_list, threshold=0.5, useFF=False):
    if gen_list == [] or ref_list == []:
        return None, None
    cov_count = 0
    mat_sum = 0
    for gen_mol in gen_list:
        if useFF is True:
            try:
                MMFFOptimizeMolecule(gen_mol)
            except:
                pass
        rmsd_list = []
        for ref_mol in ref_list:
            rmsd = GetBestRMSD(gen_mol, ref_mol)
            rmsd_list.append(rmsd)
        if min(rmsd_list) <= threshold:
            cov_count += 1
        mat_sum += min(rmsd_list)

    return 100 * cov_count / len(gen_list), mat_sum / len(gen_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="threshold of COV score"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.threshold == 0.5:
        rdkit_pkl = "" # input qm9 pickle
    else:
        rdkit_pkl = "" # input drugs pickle
    with open(rdkit_pkl, "rb") as f:
        test_dict = pickle.load(f)
    with open(args.input, "r") as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    # test_generated_data = [[[],[]] for _ in range(200)]
    test_generated_data = dict()
    for line in lines:
        gt_rdmol = test_dict[line["smi"]][0]
        if line["smi"] not in test_generated_data:
            test_generated_data[line["smi"]] = [test_dict[line["smi"]], []]

        pred_rdmol = Chem.Mol(gt_rdmol)
        pred_conformer = pred_rdmol.GetConformer()
        for atom_idx in range(pred_rdmol.GetNumAtoms()):
            new_coords = tuple(line["prediction"]["coordinates"][atom_idx])
            pred_conformer.SetAtomPosition(atom_idx, new_coords)
        test_generated_data[line["smi"]][1].append(pred_rdmol)

    print(len(test_generated_data))
    cov_list, mat_list, cov_p_list, mat_p_list = [], [], [], []
    valid_count = 0
    all_count = 0
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    smi_keys = list(test_generated_data.keys())
    for _ in tqdm(range(len(smi_keys))):
        i = smi_keys[_]
        ref_list = test_generated_data[i][0]
        gen_list = test_generated_data[i][1]
        valid_count += len(gen_list)
        all_count += len(ref_list)
        cov, mat = get_cov_mat(gen_list, ref_list, threshold=args.threshold)
        cov_list.append(cov)
        mat_list.append(mat)
        cov_p, mat_p = get_cov_mat_p(gen_list, ref_list, threshold=args.threshold)
        cov_p_list.append(cov_p)
        mat_p_list.append(mat_p)

    cov_list = list(filter(lambda x: x is not None, cov_list))
    mat_list = list(filter(lambda x: x is not None, mat_list))
    cov_p_list = list(filter(lambda x: x is not None, cov_p_list))
    mat_p_list = list(filter(lambda x: x is not None, mat_p_list))

    print(
        "Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n"
        % (
            statistics.mean(cov_list),
            statistics.median(cov_list),
            statistics.mean(mat_list),
            statistics.median(mat_list),
        )
    )
    print(
        "Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f"
        % (
            statistics.mean(cov_p_list),
            statistics.median(cov_p_list),
            statistics.mean(mat_p_list),
            statistics.median(mat_p_list),
        )
    )
    print("Valid: %.4f" % (50 * valid_count / all_count))
    print("Available: %.4f" % (100 * len(cov_list) / len(test_generated_data)))
    # save results
    with open(args.output, "a+") as f:
        f.write(f"{args.input}\n")
        f.write(
            "Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n"
            % (
                statistics.mean(cov_list),
                statistics.median(cov_list),
                statistics.mean(mat_list),
                statistics.median(mat_list),
            )
        )
        f.write(
            "Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f\n"
            % (
                statistics.mean(cov_p_list),
                statistics.median(cov_p_list),
                statistics.mean(mat_p_list),
                statistics.median(mat_p_list),
            )
        )
        f.write("Valid: %.4f\n" % (50 * valid_count / all_count))
        f.write("Available: %.4f\n" % (100 * len(cov_list) / len(test_generated_data)))

    # cov_list, mat_list, cov_p_list, mat_p_list = [], [], [], []
    # valid_count = 0
    # all_count = 0
    # logger = RDLogger.logger()
    # logger.setLevel(RDLogger.CRITICAL)
    # for _ in tqdm(range(len(smi_keys))):
    #     i = smi_keys[_]
    #     ref_list = test_generated_data[i][0]
    #     gen_list = test_generated_data[i][1]
    #     valid_count += len(gen_list)
    #     all_count += len(ref_list)
    #     cov, mat = get_cov_mat(gen_list, ref_list, threshold=args.threshold, useFF=True)
    #     cov_list.append(cov)
    #     mat_list.append(mat)
    #     cov_p, mat_p = get_cov_mat_p(gen_list, ref_list, threshold=args.threshold)
    #     cov_p_list.append(cov_p)
    #     mat_p_list.append(mat_p)

    # cov_list = list(filter(lambda x: x is not None, cov_list))
    # mat_list = list(filter(lambda x: x is not None, mat_list))
    # cov_p_list = list(filter(lambda x: x is not None, cov_p_list))
    # mat_p_list = list(filter(lambda x: x is not None, mat_p_list))

    # print("\nuseFF")
    # print(
    #     "Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f\n"
    #     % (
    #         statistics.mean(cov_list),
    #         statistics.median(cov_list),
    #         statistics.mean(mat_list),
    #         statistics.median(mat_list),
    #     )
    # )
    # print(
    #     "Coverage-Precision Mean: %.4f | Coverage-Precision Median: %.4f | Match-Precision Mean: %.4f | Match-Precision Median: %.4f"
    #     % (
    #         statistics.mean(cov_p_list),
    #         statistics.median(cov_p_list),
    #         statistics.mean(mat_p_list),
    #         statistics.median(mat_p_list),
    #     )
    # )
    # print("Valid: %.4f" % (50 * valid_count / all_count))
    # print("Available: %.4f" % (100 * len(cov_list) / len(test_generated_data)))
