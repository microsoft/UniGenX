# -*- coding: utf-8 -*-
import json
import pickle
import random
import re
import zlib
from collections import OrderedDict
from enum import Enum
from functools import cmp_to_key
from typing import List, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from unigenx.data.tokenizer import normalize_frac_coordinate
from unigenx.logging import logger


class MODE(Enum):
    TRAIN = 1
    VAL = 2
    INFER = 3


# allow pad_num to be int or float
def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N) -> (1, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def collate_fn(samples: List[dict], tokenizer, mode=MODE.TRAIN):
    """
    Overload BaseWrapperDataset.collater

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """

    max_tokens = max(len(s["tokens"]) for s in samples)
    max_masks = max(
        len(s["coordinates_mask"]) + max_tokens - len(s["tokens"]) for s in samples
    )

    batch = dict()

    if "id" in samples[0]:
        batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)

    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    batch["input_ids"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]).long(),
                max_tokens,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )

    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.padding_idx).long()

    if mode != MODE.INFER:
        batch["label_ids"] = batch["input_ids"].clone()

    if "coordinates" in samples[0]:
        batch["input_coordinates"] = torch.cat(
            [torch.from_numpy(s["coordinates"]) for s in samples]
        ).to(torch.float32)
        if mode != MODE.INFER:
            batch["label_coordinates"] = batch["input_coordinates"].clone()

    batch["coordinates_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["coordinates_mask"]).long(),
                max_masks,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )
    return batch


def normalize_frac_coordinates(coordinates: list, margin: float = 1e-4):
    return [normalize_frac_coordinate(x, margin) for x in coordinates]


def compare_by_coords(order=None):
    def innfer_f(a, b):
        frac_a = a["fractional_coordinates"]
        frac_b = b["fractional_coordinates"]
        if order == "<orderxyz>" or order is None:
            pass
        elif order == "<orderxzy>":
            frac_a = [frac_a[0], frac_a[2], frac_a[1]]
            frac_b = [frac_b[0], frac_b[2], frac_b[1]]
        elif order == "<orderyxz>":
            frac_a = [frac_a[1], frac_a[0], frac_a[2]]
            frac_b = [frac_b[1], frac_b[0], frac_b[2]]
        elif order == "<orderyzx>":
            frac_a = [frac_a[1], frac_a[2], frac_a[0]]
            frac_b = [frac_b[1], frac_b[2], frac_b[0]]
        elif order == "<orderzxy>":
            frac_a = [frac_a[2], frac_a[0], frac_a[1]]
            frac_b = [frac_b[2], frac_b[0], frac_b[1]]
        elif order == "<orderzyx>":
            frac_a = [frac_a[2], frac_a[1], frac_a[0]]
            frac_b = [frac_b[2], frac_b[1], frac_b[0]]
        else:
            raise ValueError(f"Unknown order {order}")
        if frac_a[0] > frac_b[0]:
            return 1
        elif frac_a[0] < frac_b[0]:
            return -1
        elif frac_a[1] > frac_b[1]:
            return 1
        elif frac_a[1] < frac_b[1]:
            return -1
        elif frac_a[2] > frac_b[2]:
            return 1
        elif frac_a[2] < frac_b[2]:
            return -1
        else:
            return 0

    return innfer_f


def sort_sites(sites, order=None):
    # sort the sites according to their distance to the start site
    sites_dict = OrderedDict()
    for site in sites:
        elem = site["element"]
        if elem not in sites_dict:
            sites_dict[elem] = []
        sites_dict[elem].append(site)
    for elem in sites_dict:
        sites_dict[elem] = sorted(
            sites_dict[elem],
            key=cmp_to_key(compare_by_coords(order)),
        )
    ret = []
    for elem in sites_dict:
        ret.extend(sites_dict[elem])
    return ret


class UnifiedThreeDimARGenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]] = None,
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
        material_coeff=1,
        molecule_coeff=1,
    ):
        mat_args = args.copy()
        mat_args.target = "uni_mat"
        mol_args = args.copy()
        mol_args.target = "uni_mol"
        material_data_path, molecule_data_path = data_path[0], data_path[1]
        material_dataset = ThreeDimARGenDataset(
            tokenizer, material_data_path, mat_args, shuffle=shuffle, mode=mode
        )
        molecule_dataset = ThreeDimARGenDataset(
            tokenizer, molecule_data_path, mol_args, shuffle=shuffle, mode=mode
        )
        self.material_dataset = material_dataset
        self.molecule_dataset = molecule_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.material_coeff = material_coeff
        self.molecule_coeff = molecule_coeff

    def __len__(self):
        return self.material_coeff * len(
            self.material_dataset
        ) + self.molecule_coeff * len(self.molecule_dataset)

    def __getitem__(self, idx):
        material_count = self.material_coeff * len(self.material_dataset)
        if idx < material_count:
            material_idx = idx % len(self.material_dataset)
            return self.material_dataset[material_idx]
        else:
            molecule_idx = (idx - material_count) % len(self.molecule_dataset)
            return self.molecule_dataset[molecule_idx]

    def collate(self, samples):
        return collate_fn(samples, self.tokenizer, self.mode)


class ThreeDimARGenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]] = None,
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.max_position_embeddings = args.max_position_embeddings

        self.data = []
        self.sizes = []
        self.env = None
        self.keys = None

        if data_path is not None:
            if data_path.count(",") > 0:
                data_path = data_path.split(",")
            if isinstance(data_path, str):
                data_path = [data_path]
            for path in data_path:
                self.load_data_from_file(path)

        assert (self.data == [] and self.keys is not None) or (
            self.data != [] and self.keys is None
        )

        if shuffle:
            if self.keys is not None:
                cb = list(zip(self.sizes, self.keys))
                random.shuffle(cb)
                self.sizes, self.keys = zip(*cb)
                self.sizes = list(self.sizes)
                self.keys = list(self.keys)
            else:
                random.shuffle(self.data)

        if self.args.scale_coords:
            logger.info(f"scale coords with scale {self.args.scale_coords}")

    def get_sequence_length(self, data_item):
        if self.args.target == "material" or self.args.target == "uni_mat":
            # <bos> [n * ] <coords> [3 lattice] [n coords] <eos>
            n = len(data_item["sites"])
            return 1 + n + 1 + 3 + n + 1
        elif self.args.target == "mol" or self.args.target == "uni_mol":
            # <bos> smiles <coords> [n coords] <eos>
            return len(data_item["smi"]) + 3 + data_item["num"]
        elif self.args.target == "prot":
            raise NotImplementedError
        elif self.args.target == "cond_mol":
            return 3
        else:
            return 0

    def load_dict(self, lines: List[dict]):
        skipped = 0
        for data_item in lines:
            size = self.get_sequence_length(data_item)
            if (
                self.args.target == "material"
                and self.args.max_sites is not None
                and len(data_item["sites"]) > self.args.max_sites
            ):
                skipped += 1
                continue
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue

            if self.mode != MODE.INFER and self.args.target == "material":
                # normalize fractional coordinates
                for i in range(len(data_item["sites"])):
                    data_item["sites"][i][
                        "fractional_coordinates"
                    ] = normalize_frac_coordinates(
                        data_item["sites"][i]["fractional_coordinates"]
                    )
                sorted_sites = sort_sites(data_item["sites"])
                data_item["sites"] = sorted_sites

            self.data.append(data_item)  # type(data_item:) = dict
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def load_json(self, lines: List[str]):
        lines = [json.loads(line) for line in lines]
        self.load_dict(lines)

    def load_txt(self, lines: List[str]):
        skipped = 0
        for line in lines:
            data_item = line.strip()
            size = self.get_sequence_length(data_item)
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue
            self.data.append(data_item)
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def infer_data_format(self, data_path, data_format):
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            file_format = "json"
        elif data_path.endswith(".txt"):
            file_format = "txt"
        elif data_path.endswith(".lmdb"):
            file_format = "lmdb"
        elif data_path.endswith("pickle") or data_path.endswith("pkl"):
            file_format = "pickle"
        else:
            raise ValueError(f"Unknown data format {data_path}")
        if data_format is not None:
            if data_format == file_format:
                return file_format
            else:
                return data_format
        return file_format

    def load_data_from_file(self, data_path, data_format=None):
        data_path = data_path.strip()
        data_format = self.infer_data_format(data_path, data_format)

        if data_format == "json":
            with open(data_path, "r") as f:
                lines = f.readlines()
            self.load_json(lines)
        elif data_format == "txt":
            with open(data_path, "r") as f:
                lines = f.readlines()
            self.load_txt(lines)
        elif data_format == "lmdb":
            self.env = lmdb.open(data_path, readonly=True, lock=False, readahead=False)
            self.txn = self.env.begin(write=False)
            try:
                metadata = pickle.loads(
                    zlib.decompress(self.txn.get("__metadata__".encode()))
                )
                self.sizes, self.keys = metadata["sizes"], metadata["keys"]
            except:
                cursor = self.txn.cursor()
                for key, _ in cursor:
                    self.keys.append(key)
        elif data_format == "pickle":
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raise ValueError(f"Unknown data format {data_format}")

    def __len__(self):
        if len(self.data) != 0:
            return len(self.data)
        elif self.keys is not None:
            return len(self.keys)
        else:
            raise ValueError("Dataset is empty")

    def get_infer_item_mat(self, index):
        item = dict()
        data_item = self.data[index]

        sorted_sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend(
            [self.tokenizer.get_idx(site["element"]) for site in sorted_sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sorted_sites))])

        if self.args.space_group:
            # add special token
            sites_ids.append(self.tokenizer.coord_idx)
            coordinates_mask.append(0)

            # add order if needed
            if self.args.reorder:
                sites_ids.append(self.tokenizer.get_idx(self.tokenizer.order_tokens[0]))
                coordinates_mask.append(0)
        else:
            """
            # mask for space group
            coordinates_mask.append(0)
            # mask for special token
            coordinates_mask.append(0)
            """
            sites_ids.append(self.tokenizer.coord_idx)
            coordinates_mask.append(0)
            # mask for order
            if self.args.reorder:
                coordinates_mask.append(0)

        # add mask for lattice
        coordinates_mask.extend([1 for _ in range(3)])

        # add mask for coordinates
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        atom_num = data_item["num"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))

        sites_ids.extend(smiles_id)
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        coordinates_mask.extend([1 for _ in range(atom_num)])

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_prot(self, index):
        if self.env is not None:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.keys[index].encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = self.data[index]

        item = dict()
        seq = data_item.get("seq", data_item.get("aa"))

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        sites_ids.extend([self.tokenizer.get_idx(res) for res in seq])
        coordinates_mask.extend([0 for _ in range(len(seq))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # add coordinates
        coordinates_mask.extend([1 for _ in range(len(seq))])

        # eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_cond_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        prop = data_item["prop"]
        sites_ids.extend(
            [
                self.tokenizer.get_idx(f"<{prop}>"),
                self.tokenizer.mask_idx,
            ]
        )
        coordinates_mask.extend([0, 1])

        prop_val = data_item["prop_val"]
        coordinates = np.array([prop_val, prop_val, prop_val]).reshape(1, 3)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_cond_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        prop_val = data_item["prop_val"]
        prop = data_item["prop"]

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # property name
        sites_ids.append(self.tokenizer.get_idx(f"<{prop}>"))
        coordinates_mask.append(0)

        # property value
        sites_ids.append(self.tokenizer.mask_idx)
        coordinates_mask.append(1)

        # begin of words
        sites_ids.append(self.tokenizer.get_idx("<w>"))
        coordinates_mask.append(0)

        coordinates = np.array([prop_val, prop_val, prop_val]).reshape(1, 3)
        coordinates_mask = np.array(coordinates_mask)
        sites_ids = np.array(sites_ids)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_uni_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<material>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        # get all sites
        sorted_sites = data_item["sites"]
        sites_ids.extend(
            [self.tokenizer.get_idx(f'<m>{site["element"]}') for site in sorted_sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sorted_sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # mask for order
        if self.args.reorder:
            coordinates_mask.append(0)

        # add mask for lattice
        coordinates_mask.extend([1 for _ in range(3)])

        # add mask for coordinates
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_uni_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]
        # data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<molecule>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        smiles = data_item["smi"]
        atom_num = data_item["num"]
        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(
                    f"<s>{smiles[i - 1] + smiles[i]}"
                )
            else:
                smiles_id.append(self.tokenizer.get_idx(f"<s>{smiles[i]}"))
        # print(smiles)
        sites_ids.extend(smiles_id)
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])
        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # add coordinates
        coordinates_mask.extend([1 for _ in range(atom_num)])

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item.get("id", index)
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add space group
        # By zgb: no space group test
        """
        sites_ids.append(self.tokenizer.sg_idx)
        coordinates_mask.append(0)
        space_group_no = str(data_item["space_group"]["no"])
        space_group_tok = f"<sgn>{space_group_no}"
        sites_ids.append(self.tokenizer.get_idx(space_group_tok))
        coordinates_mask.append(0)
        """

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item
    
    def get_train_item_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        coords = data_item["pos"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item    

    def get_train_item_prot(self, index):
        item = dict()
        key = self.data[index]
        if isinstance(key, str):
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(key.encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = key
        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # tokenize sequence
        seq = data_item["aa"]
        sites_ids.extend([self.tokenizer.get_idx(char) for char in seq])
        coordinates_mask.extend([0 for _ in range(len(seq))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        coordinates = data_item["pos"]

        coord_mean = np.mean(coordinates, axis=0)
        coordinates = coordinates - coord_mean  # 取中心

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(seq))])
        coordinates_mask.extend([1 for _ in range(len(seq))])

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item
    
    def get_train_cond_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        coords = data_item["pos"]
        prop_val = data_item["prop_val"]
        prop = data_item["prop"]

        # property name
        sites_ids.append(self.tokenizer.get_idx(f"<{prop}>"))
        coordinates_mask.append(0)

        # property value
        sites_ids.append(self.tokenizer.mask_idx)
        coordinates_mask.append(1)

        # begin of words
        sites_ids.append(self.tokenizer.get_idx("<w>"))
        coordinates_mask.append(0)

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        coordinates = np.insert(
            coordinates, 0, np.array([prop_val, prop_val, prop_val]), axis=0
        )

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_cond_mat(self, index):
        from math import exp, log

        item = dict()
        tags = []
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        sites_ids = []
        coordinates_mask = []

        # property
        for key in data_item["property"]:
            prop = re.search(r"dft_(.*?)_", key)
            if prop:
                prop_tok = prop.group(1)
                sites_ids.extend(
                    [
                        self.tokenizer.get_idx(f"<{prop_tok}>"),
                        self.tokenizer.mask_idx,
                    ]
                )
                coordinates_mask.extend([0, 1])
                tags.append(0)  # diffloss on property
                prop_val = data_item["property"][key]
                if prop_tok == "bulk":
                    if prop_val > 0:
                        prop_val = log(prop_val + 1)
                    else:
                        prop_val = -log(-prop_val + 1)
                elif prop_tok == "band":
                    prop_val = log(prop_val + 1 / exp(1))
                else:  # mag
                    if prop_val > 0:
                        prop_val = -log(prop_val) / 10
                    elif prop_val < 0:
                        prop_val = log(-prop_val) / 10
                break

        # begin with bos
        sites_ids.append(self.tokenizer.bos_idx)
        coordinates_mask.append(0)

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])
        tags.extend([0 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        tags.extend([0 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        tags = np.array(tags)

        coordinates = np.insert(
            coordinates, 0, np.array([prop_val, prop_val, prop_val]), axis=0
        )
        assert len(tags) == len(coordinates), f"{len(tags)}, {len(coordinates)}"
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        # toks = []
        # for i in sites_ids:
        #     toks.append(self.tokenizer.get_tok(i))
        # print(toks)
        # print(coordinates)
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["tags"] = tags
        return item

    def get_train_uni_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<material>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        # get all sites
        sites_ids.extend(
            [self.tokenizer.get_idx(f'<m>{site["element"]}') for site in sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_uni_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<molecule>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        smiles = data_item["smi"]
        coords = data_item["pos"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(
                    f"<s>{smiles[i - 1] + smiles[i]}"
                )
            else:
                smiles_id.append(self.tokenizer.get_idx(f"<s>{smiles[i]}"))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item(self, index):
        if self.args.target == "material":
            return self.get_train_item_mat(index)
        elif self.args.target == "mol":
            return self.get_train_item_mol(index)
        elif self.args.target == "prot":
            return self.get_train_item_prot(index)
        elif self.args.target == "cond_mat":
            return self.get_train_cond_mat(index)
        elif self.args.target == "cond_mol":
            return self.get_train_cond_mol(index)
        elif self.args.target == "uni_mat":
            return self.get_train_uni_mat(index)
        elif self.args.target == "uni_mol":
            return self.get_train_uni_mol(index)
        else:
            raise ValueError(f"Unknown target {self.args.target}")

    def get_infer_item(self, index):
        if self.args.target == "material":
            return self.get_infer_item_mat(index)
        elif self.args.target == "mol":
            return self.get_infer_item_mol(index)
        elif self.args.target == "prot":
            return self.get_infer_item_prot(index)
        elif self.args.target == "cond_mat":
            return self.get_infer_cond_mat(index)
        elif self.args.target == "cond_mol":
            return self.get_infer_cond_mol(index)
        elif self.args.target == "uni_mat":
            return self.get_infer_uni_mat(index)
        elif self.args.target == "uni_mol":
            return self.get_infer_uni_mol(index)
        else:
            raise ValueError(f"Unknown target {self.args.target }")

    def __getitem__(self, index):
        if self.mode in [MODE.TRAIN, MODE.VAL]:
            return self.get_train_item(index)
        elif self.mode == MODE.INFER:
            return self.get_infer_item(index)

    def _random_rotation(self, lattice):
        # Generate random rotation angles
        angles = np.random.uniform(0, 2 * np.pi, size=3)

        # Compute sine and cosine of angles
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)

        # Construct rotation matrix
        rotation_matrix = np.eye(3)

        rotation_matrix[0, 0] = cos_angles[0] * cos_angles[1]
        rotation_matrix[0, 1] = (
            cos_angles[0] * sin_angles[1] * sin_angles[2]
            - sin_angles[0] * cos_angles[2]
        )
        rotation_matrix[0, 2] = (
            cos_angles[0] * sin_angles[1] * cos_angles[2]
            + sin_angles[0] * sin_angles[2]
        )
        rotation_matrix[1, 0] = sin_angles[0] * cos_angles[1]
        rotation_matrix[1, 1] = (
            sin_angles[0] * sin_angles[1] * sin_angles[2]
            + cos_angles[0] * cos_angles[2]
        )
        rotation_matrix[1, 2] = (
            sin_angles[0] * sin_angles[1] * cos_angles[2]
            - cos_angles[0] * sin_angles[2]
        )
        rotation_matrix[2, 0] = -sin_angles[1]
        rotation_matrix[2, 1] = cos_angles[1] * sin_angles[2]
        rotation_matrix[2, 2] = cos_angles[1] * cos_angles[2]

        # Rotate lattice
        lattice = np.dot(rotation_matrix, lattice.T).T

        return lattice

    def collate(self, samples):
        return collate_fn(samples, self.tokenizer, self.mode)
