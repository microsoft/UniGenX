# -*- coding: utf-8 -*-
# import os
# import pickle
# import argparse
# import torch
# import numpy as np
# from psikit import Psikit
# from tqdm.auto import tqdm
# from easydict import EasyDict
# from copy import deepcopy
# from collections import defaultdict
# import json
# from rdkit import Chem
# from rdkit.Chem import AllChem

# def set_rdmol_positions(rdkit_mol, pos):
#     """
#     Args:
#         rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
#         pos: (N_atoms, 3)
#     """
#     mol = deepcopy(rdkit_mol)
#     set_rdmol_positions_(mol, pos)
#     return mol


# def set_rdmol_positions_(mol, pos):
#     """
#     Args:
#         rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
#         pos: (N_atoms, 3)
#     """
#     for i in range(pos.shape[0]):
#         mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
#     return mol

# def pack_data_by_mol(data):
#         """
#         pack confs with same mol into a single data object
#         """
#         packed_data = defaultdict(list)
#         if hasattr(data, 'idx'):
#             for i in range(len(data)):
#                 packed_data[data[i].idx.item()].append(data[i])
#         else:
#             for i in range(len(data)):
#                 packed_data[data[i].smiles].append(data[i])
#         print('[Packed] %d Molecules, %d Conformations.' % (len(packed_data), len(data)))

#         new_data = []
#         # logic
#         # save graph structure for each mol once, but store all confs
#         for k, v in packed_data.items():
#             data = deepcopy(v[0])
#             all_pos = []
#             for i in range(len(v)):
#                 all_pos.append(v[i].pos)
#             data.pos_ref = torch.cat(all_pos, 0) # (num_conf*num_node, 3)
#             data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
#             #del data.pos

#             if hasattr(data, 'totalenergy'):
#                 del data.totalenergy
#             if hasattr(data, 'boltzmannweight'):
#                 del data.boltzmannweight
#             new_data.append(data)

#         return new_data

# class PropertyCalculator(object):

#     def __init__(self, threads, memory, seed):
#         super().__init__()
#         self.pk = Psikit(threads=threads, memory=memory)
#         self.seed = seed

#     def __call__(self, data, num_confs=50):
#         rdmol = data.rdmol
#         confs = data.pos_prop

#         conf_idx = np.arange(confs.shape[0])
#         np.random.RandomState(self.seed).shuffle(conf_idx)
#         conf_idx = conf_idx[:num_confs]

#         data.prop_conf_idx = []
#         data.prop_energy = []
#         data.prop_homo = []
#         data.prop_lumo = []
#         data.prop_dipo = []

#         for idx in tqdm(conf_idx):
#             mol = set_rdmol_positions(rdmol, confs[idx])
#             self.pk.mol = mol
#             try:
#                 energy, homo, lumo, dipo = self.pk.energy(), self.pk.HOMO, self.pk.LUMO, self.pk.dipolemoment[-1]
#                 data.prop_conf_idx.append(idx)
#                 data.prop_energy.append(energy)
#                 data.prop_homo.append(homo)
#                 data.prop_lumo.append(lumo)
#                 data.prop_dipo.append(dipo)
#             except:
#                 pass

#         return data


# def get_prop_matrix(data):
#     """
#     Returns:
#         properties: (4, num_confs) numpy tensor. Energy, HOMO, LUMO, DipoleMoment
#     """
#     return np.array([
#         data.prop_energy,
#         data.prop_homo,
#         data.prop_lumo,
#         data.prop_dipo,
#     ])


# def get_ensemble_energy(props):
#     """
#     Args:
#         props: (4, num_confs)
#     """
#     avg_ener = np.mean(props[0, :])
#     low_ener = np.min(props[0, :])
#     gaps = np.abs(props[1, :] - props[2, :])
#     avg_gap = np.mean(gaps)
#     min_gap = np.min(gaps)
#     max_gap = np.max(gaps)
#     return np.array([
#         avg_ener, low_ener, avg_gap, min_gap, max_gap,
#     ])

# HART_TO_EV = 27.211
# HART_TO_KCALPERMOL = 627.5

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='')
#     parser.add_argument('--generated', type=str, default='')
#     parser.add_argument('--num_confs', type=int, default=50)
#     parser.add_argument('--threads', type=int, default=8)
#     parser.add_argument('--memory', type=int, default=16)
#     parser.add_argument('--seed', type=int, default=2021)
#     args = parser.parse_args()

#     prop_cal = PropertyCalculator(threads=args.threads, memory=args.memory, seed=args.seed)

#     cache_ref_fn = os.path.join(
#         os.path.dirname(args.dataset),
#         os.path.basename(args.dataset)[:-4] + '_prop.pkl'
#     )
#     if not os.path.exists(cache_ref_fn):
#         with open(args.dataset, 'rb') as f:
#             dset = pickle.load(f)
#         dset = pack_data_by_mol(dset)
#         dset_prop = []
#         for data in dset:
#             data.pos_prop = data.pos_ref.reshape(-1, data.num_nodes, 3)
#             dset_prop.append(prop_cal(data, args.num_confs))
#         with open(cache_ref_fn, 'wb') as f:
#             pickle.dump(dset_prop, f)
#         dset = dset_prop
#     else:
#         with open(cache_ref_fn, 'rb') as f:
#             dset = pickle.load(f)


#     if args.generated is None:
#         exit()

#     print('Start evaluation.')

#     cache_gen_fn = os.path.join(
#         os.path.dirname(args.generated),
#         os.path.basename(args.generated)[:-4] + '_prop.pkl'
#     )
#     if True:
#         with open(args.generated, 'r') as f:
#             gens = f.readlines()
#         gens = [json.loads(line) for line in gens]
#         gens = [gens[i:i + 50] for i in range(0, len(gens), 50)]
#         with open('', 'rb') as f:
#             rd_dict = pickle.load(f)
#         gens_prop = []
#         for chunk in gens:
#             data = deepcopy(chunk[0])
#             data = EasyDict(data)
#             all_pos = []
#             for i in range(50):
#                 all_pos.append(torch.tensor(chunk[i]['prediction']['coordinates']))
#             a = torch.cat(all_pos, 0)
#             data.rdmol = rd_dict[data.smi]
#             data.num_nodes = data.rdmol.GetNumAtoms()
#             data.pos_prop = a.reshape(-1, data.num_nodes, 3)
#             gens_prop.append(prop_cal(data, args.num_confs))
#         with open(cache_gen_fn, 'wb') as f:
#             pickle.dump(gens_prop, f)
#         gens = gens_prop
#     else:
#         with open(cache_gen_fn, 'rb') as f:
#             gens = pickle.load(f)


#     all_diff = []

#     for d, g in zip(dset, gens):
#         # if smiles not in gens:
#         #     continue
#         if not g.prop_energy:
#             continue

#         prop_gts = get_ensemble_energy(get_prop_matrix(d)) * HART_TO_EV
#         prop_gen = get_ensemble_energy(get_prop_matrix(g)) * HART_TO_EV
#         # prop_gts = np.mean(get_prop_matrix(dset[smiles]), axis=1)
#         # prop_gen = np.mean(get_prop_matrix(gens[smiles]), axis=1)

#         # print(get_prop_matrix(gens[smiles])[0])

#         prop_diff = np.abs(prop_gts - prop_gen)

#         print('\nProperty: %s' % g.smi)
#         print('  Gts :', prop_gts)
#         print('  Gen :', prop_gen)
#         print('  Diff:', prop_diff)

#         all_diff.append(prop_diff.reshape(1, -1))
#     all_diff = np.vstack(all_diff)  # (num_mols, 4)
#     print(all_diff.shape)

#     print('[Difference]')
#     print('  Mean:  ', np.mean(all_diff, axis=0))
#     print('  Median:', np.median(all_diff, axis=0))
#     print('  Std:   ', np.std(all_diff, axis=0))

import pickle

with open(" ", "rb") as f:
    final_test = pickle.load(f)

import numpy as np
from psikit import Psikit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

pk = Psikit(threads=8, memory=16)
HART_TO_EV = 27.211
all_diff = []
save_list = []
for t in tqdm(final_test):
    gt_energy = []
    gt_gap = []
    for m in tqdm(t[0]):
        try:
            mol_with_h = Chem.AddHs(m, addCoords=True)
            pk.mol = mol_with_h
            e, g = pk.energy(), pk.HOMO - pk.LUMO
            gt_energy.append(float(e))
            gt_gap.append(abs(float(g)))
        except:
            pass
    pred_energy = []
    pred_gap = []
    for m in tqdm(t[1]):
        try:
            mol_with_h = Chem.AddHs(m, addCoords=True)
            pk.mol = mol_with_h
            e, g = pk.energy(), pk.HOMO - pk.LUMO
            pred_energy.append(float(e))
            pred_gap.append(abs(float(g)))
        except:
            pass
    save_list.append([gt_energy, pred_energy, gt_gap, pred_gap])
    gt_energy_mean = sum(gt_energy) / len(gt_energy)
    gt_energy_min = min(gt_energy)
    gt_gap_mean = sum(gt_gap) / len(gt_gap)
    gt_gap_min = min(gt_gap)
    gt_gap_max = max(gt_gap)
    prop_gts = (
        np.array([gt_energy_mean, gt_energy_min, gt_gap_mean, gt_gap_min, gt_gap_max])
        * HART_TO_EV
    )
    pred_energy_mean = sum(pred_energy) / len(pred_energy)
    pred_energy_min = min(pred_energy)
    pred_gap_mean = sum(pred_gap) / len(pred_gap)
    pred_gap_min = min(pred_gap)
    pred_gap_max = max(pred_gap)
    prop_gen = (
        np.array(
            [
                pred_energy_mean,
                pred_energy_min,
                pred_gap_mean,
                pred_gap_min,
                pred_gap_max,
            ]
        )
        * HART_TO_EV
    )
    prop_diff = np.abs(prop_gts - prop_gen)
    print(prop_diff)
    all_diff.append(prop_diff.reshape(1, -1))

with open("save_prop.pkl", "wb") as f:
    pickle.dump(save_list, f)
all_diff = np.vstack(all_diff)

print("[Difference]")
print("  Mean:  ", np.mean(all_diff, axis=0))
print("  Median:", np.median(all_diff, axis=0))
print("  Std:   ", np.std(all_diff, axis=0))
