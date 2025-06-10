# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np


def flatten_formula(formula):
    output = []

    i = 0
    while i < len(formula):
        char = formula[i]

        if char.isupper():
            elem_start = i
            i += 1
            while i < len(formula) and formula[i].islower():
                i += 1
            elem = formula[elem_start:i]

            if i < len(formula) and formula[i].isdigit():
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                num = int(formula[num_start:i])
            else:
                num = 1

            output.extend([elem] * num)

        elif char == "(":
            group_elem = []
            group_elem_num = []
            i += 1

            while i < len(formula) and formula[i] != ")":
                if formula[i].isupper():
                    elem_start = i
                    i += 1
                    while (
                        i < len(formula) and formula[i] != ")" and formula[i].islower()
                    ):
                        i += 1
                    elem = formula[elem_start:i]

                if i < len(formula) and formula[i] != ")" and formula[i].isdigit():
                    num_start = i
                    while (
                        i < len(formula) and formula[i] != ")" and formula[i].isdigit()
                    ):
                        i += 1
                    num = int(formula[num_start:i])
                else:
                    num = 1
                group_elem.append(elem)
                group_elem_num.append(num)
            i += 1
            if i < len(formula) and formula[i].isdigit():
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                group_num = int(formula[num_start:i])
            else:
                group_num = 1

            for elem, num in zip(group_elem, group_elem_num):
                output.extend([elem] * num * group_num)

    return output


def normalize_frac_coordinate(x: float, margin: float = 1e-4):
    if x < 0:
        x = x + abs(int(x)) + 1
    if x > 1:
        x = x - int(x)
    # adjust value near 0 or 1 to 0
    if min(abs(x - 0), abs(x - 1)) < margin:
        x = float(0.0)
    x = round(x, 6)
    return x


class UniGenXTokenizer(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Dict[str, str] = OrderedDict(
            {
                "pad": "<pad>",
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
            }
        ),
        append_toks: Dict[str, str] = OrderedDict(
            {
                "mask": "<mask>",
                "coord": "<coord>",
            }
        ),
        args=None,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks.values())
        self.append_toks = list(append_toks.values())

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        for key, value in prepend_toks.items():
            setattr(self, f"{key}_tok", value)
            setattr(self, f"{key}_idx", self.tok_to_idx[value])
        for key, value in append_toks.items():
            setattr(self, f"{key}_tok", value)
            setattr(self, f"{key}_idx", self.tok_to_idx[value])
        setattr(self, "padding_idx", self.tok_to_idx[self.pad_tok])

        self.args = args

        self.reorder = getattr(args, "reorder", False)
        if self.reorder:
            self.order_tokens = [
                "<orderxyz>",
                "<orderxzy>",
                "<orderyxz>",
                "<orderyzx>",
                "<orderzxy>",
                "<orderzyx>",
            ]
            self.order_token_ids = [self.add_tok(token) for token in self.order_tokens]

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def add_tok(self, tok):
        if tok in self.tok_to_idx:
            return self.tok_to_idx[tok]
        self.all_toks.append(tok)
        self.tok_to_idx[tok] = len(self.all_toks) - 1
        return self.tok_to_idx[tok]

    def tokenize(
        self, text, prepend_bos=True, append_eos=False, append_gen=False
    ) -> List[str]:
        res = flatten_formula(text)
        if prepend_bos:
            res = [self.bos_tok] + res
        if append_gen:
            res = res + [self.coord_tok]
        if append_eos:
            res = res + [self.eos_tok]
        return res

    def encode(self, text, prepend_bos=True, append_eos=False, append_gen=False):
        return np.array(
            [
                self.get_idx(tok)
                for tok in self.tokenize(text, prepend_bos, append_eos, append_gen)
            ]
        )

    def decode(self, tokens, coordinates, mask, entity) -> str:
        scale_coords = getattr(self.args, "scale_coords", None)
        seq_len = tokens.shape[0]
        mask = mask[:seq_len]
        sent = []
        lattice = []
        atom_coordinates = []
        coordinates_index = 0
        for i in range(seq_len):
            if mask[i] == 1:
                x, y, z = coordinates[coordinates_index]
                if (entity == "material" or entity == "uni_mat") and coordinates_index < 3:
                    lattice.append([x, y, z])
                else:
                    atom_coordinates.append([x, y, z])
                sent.extend(map(str, [x, y, z]))
                coordinates_index += 1
            else:
                if tokens[i] not in [self.bos_idx, self.eos_idx, self.padding_idx]:
                    sent.append(self.get_tok(tokens[i]))
                if tokens[i] == self.eos_idx:
                    break
        sent = " ".join(sent)
        if scale_coords:
            atom_coordinates = [
                [x / scale_coords for x in vec] for vec in atom_coordinates
            ]
        return sent, lattice, atom_coordinates

    def decode_batch(self, tokens, coordindates, masks, entity="material") -> List[str]:
        ret = []
        bs = tokens.shape[0]
        coords_start = 0
        for i in range(bs):
            num_coords = np.sum(masks[i] != 0)
            sent, lattice, atom_coordinates = self.decode(
                tokens[i],
                coordindates[coords_start : coords_start + num_coords],
                masks[i],
                entity,
            )
            if entity == "material" or entity == "uni_mat":
                ret.append((sent, lattice, atom_coordinates))
            elif entity == "mol" or entity == "cond_mol" or entity == "uni_mol":
                ret.append((sent, atom_coordinates))
            else:
                raise ValueError(f"entity {entity} not supported")
            coords_start += num_coords
        return ret

    def get_ele_num(self, text):
        res = flatten_formula(text)
        return res

    @classmethod
    def from_file(cls, filename, args=None):
        tokens = []
        with open(filename, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split()[0]
                tokens.append(token)

        append_toks = OrderedDict(
            {
                "mask": "<mask>",
                "coord": "<coord>",
                "sg": "<sg>",
            }
        )

        return UniGenXTokenizer(tokens, append_toks=append_toks, args=args)
