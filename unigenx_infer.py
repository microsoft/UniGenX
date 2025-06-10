# -*- coding: utf-8 -*-
import json
from dataclasses import asdict
import os

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed
from transformers.generation.configuration_utils import GenerationConfig
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm

from unigenx.data.dataset import MODE, ThreeDimARGenDataset, pad_1d_unsqueeze
from unigenx.data.tokenizer import UniGenXTokenizer
from unigenx.logging import logger
from unigenx.model.config import UniGenConfig, UniGenInferenceConfig, UniGenInferencedenovoConfig
from unigenx.model.wrapper import UniGenX
from unigenx.utils import arg_utils
from unigenx.utils.cli_utils import cli
from unigenx.utils.move_to_device import move_to_device
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

SPECIAL_TOKEN_IDS = {
    "bos": None,
    "eos": None,
    "padding": None,
    "coord": None
}

JSON_SERIALIZABLE_TYPES = (np.float32,)

def convert_json_serializable(obj):
    if isinstance(obj, JSON_SERIALIZABLE_TYPES):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

@cli(UniGenConfig, UniGenInferenceConfig)
def main(args):
    # region initial config--------
    set_seed(args.seed)
    logger.info(f"Initializing with seed: {args.seed}")

    config = arg_utils.from_args(args, UniGenConfig)
    inference_config = arg_utils.from_args(args, UniGenInferenceConfig)

    if inference_config.input_file is None:
        inference_config = arg_utils.from_args(args, UniGenInferencedenovoConfig)

    checkpoints_state = torch.load(config.loadcheck_path, map_location="cpu")
    saved_args = checkpoints_state["args"]

    saved_config = arg_utils.from_args(saved_args, UniGenConfig)
    saved_config.tokenizer = "num" 
    saved_config.diff_steps = config.diff_steps
    saved_config.target = config.target
    ## modify for dpm solver ##
    saved_config.is_solver = config.is_solver
    saved_config.solver_order = config.solver_order
    saved_config.solver_type = config.solver_type
    
    for k, v in asdict(config).items():
        if not hasattr(saved_config, k):
            setattr(saved_config, k, getattr(config, k))
    saved_config.update(asdict(inference_config))
    # endregion ---------------

    # region initial model --------
    logger.info(f"Loading tokenizer from {args.dict_path}")
    tokenizer = UniGenXTokenizer.from_file(args.dict_path, saved_config)

    SPECIAL_TOKEN_IDS.update({
        "bos": tokenizer.bos_idx,
        "eos": tokenizer.eos_idx,
        "padding": tokenizer.padding_idx,
        "coord": tokenizer.coord_idx
    })

    model = UniGenX(saved_config)
    model.eval()

    logger.info(f"Loading model from {args.loadcheck_path}")
    model.load_pretrained_weights(args.loadcheck_path)
    model.cuda()
    # endregion ---------------

    # region data&GenConfig ---------------
    logger.info(f"Loading inference data from {args.input_file}")
    saved_config.mask_token_id = tokenizer.mask_idx
    if inference_config.input_file is not None:
        gen_config = GenerationConfig(
            pad_token_id=SPECIAL_TOKEN_IDS["padding"],
            eos_token_id=SPECIAL_TOKEN_IDS["eos"],
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )
        # Here we use sampling method to generate words.
        sample_config = GenerationConfig(
            pad_token_id=SPECIAL_TOKEN_IDS["padding"],
            eos_token_id=SPECIAL_TOKEN_IDS["coord"],  # Use coord_idx as the END OF SENTENCE token
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )

        dataset = ThreeDimARGenDataset(
            tokenizer,
            args.input_file,
            saved_config,
            shuffle=False,
            mode=MODE.INFER,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.infer_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
        )

        # endregion ---------------
        
        # region infer loop -------
        index = 0 # index of the currently processed data
        logger.info(f"Starting generation process...")
        with open(args.output_file, "w") as fw:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = move_to_device(batch, "cuda")

                    if args.target in {"material", "mol", "uni_mat", "uni_mol"}:
                        # Standard Gen
                        ret = model.net.generate(
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=gen_config,
                            max_length=batch["coordinates_mask"].shape[1],
                        )
                        coordinates_mask = batch["coordinates_mask"]

                    elif args.target == "cond_mol":
                        # Two steps for conditional Gen
                        # Phase1 generate initial sequence
                        ret = model.net.generate(   # Phase1 result
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=sample_config,
                            input_coordinates=batch["input_coordinates"],
                            only_seq=True,
                            max_length=dataset.max_position_embeddings // 2,
                            do_sample=True,
                            top_p=0.8,
                            temperature=0.6,
                        )

                        tokens = ret.sequences.cpu().numpy()
                        input_coordinates_batch = batch["input_coordinates"].cpu().numpy()
                        batchsize = tokens.shape[0]
                        valid_tokens = []
                        atom_nums = []
                        skip_index_list = []
                        smiles = []
                        processed_tokens = []
                        skip_index = index
                        input_coordinates_filtered = []
                        assert batchsize == batch["input_coordinates"].shape[0]
                        for batch_idx in range(batchsize):
                            sentence = []
                            if SPECIAL_TOKEN_IDS["coord"] in tokens[batch_idx]:
                                for token_idx in range(len(tokens[batch_idx])):
                                    if tokens[batch_idx][token_idx] not in [
                                        SPECIAL_TOKEN_IDS["bos"],
                                        SPECIAL_TOKEN_IDS["eos"],
                                        SPECIAL_TOKEN_IDS["padding"],
                                    ]:
                                        sentence.append(tokenizer.get_tok(tokens[batch_idx][token_idx]))
                                    if tokens[batch_idx][token_idx] == SPECIAL_TOKEN_IDS["coord"]:
                                        break
                                mol = Chem.MolFromSmiles(
                                    "".join(sentence[3:-1]), sanitize=False
                                )
                                if mol is not None:
                                    atom_nums.append(mol.GetNumAtoms())
                                    processed_tokens.append(sentence)
                                    input_coordinates_filtered.append(
                                        input_coordinates_batch[batch_idx]
                                    )
                                    token = tokens[batch_idx]
                                    valid_tokens.append(
                                        token[ : np.where(token == SPECIAL_TOKEN_IDS["coord"])[0][0] + 1].tolist()
                                    )
                                else:
                                    skip_index_list.append(skip_index)
                            else:
                                skip_index_list.append(skip_index)
                            skip_index += 1                   

                        smiles = [sentence[3:-1] for sentence in processed_tokens]
                        if len(smiles) == 0:
                            index += batchsize
                            continue
                        # prepare phase2 input
                        origin_coordinates_mask = []
                        for token_seq, atom_count in zip(valid_tokens, atom_nums):
                            mask = [0, 0, 1] + [0]*(len(token_seq)-3) + [1]*atom_count + [0]
                            origin_coordinates_mask.append(mask)

                        max_tokens = max(len(token) for token in valid_tokens)
                        max_masks = max(
                            len(origin_coordinates_mask[i]) + max_tokens - len(valid_tokens[i])
                            for i in range(len(origin_coordinates_mask))
                        )
                        input_ids = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(token).long(),
                                    max_tokens,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for token in valid_tokens
                            ]
                        )
                        coordinates_mask = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(mask).long(),
                                    max_masks,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for mask, token in zip(origin_coordinates_mask, valid_tokens)
                            ]
                        )
                        input_coordinates_filtered = torch.cat(
                            [
                                torch.from_numpy(s).unsqueeze(0)
                                for s in input_coordinates_filtered
                            ]
                        ).to(torch.float32)
                        input_ids = move_to_device(input_ids, "cuda")
                        coordinates_mask = move_to_device(coordinates_mask, "cuda")
                        input_coordinates_filtered = move_to_device(
                            input_coordinates_filtered, "cuda"
                        )
                        #Phase2 Generation
                        ret = model.net.generate(
                            input_ids=input_ids,
                            coordinates_mask=coordinates_mask,
                            input_coordinates=input_coordinates_filtered,
                            do_sample=False,
                            generation_config=gen_config,
                            max_length=coordinates_mask.shape[1],
                        )

                    # region Result processing -------
                    decoded_results = tokenizer.decode_batch(
                        ret.sequences.cpu().numpy(),
                        ret.coordinates.cpu().numpy(),
                        coordinates_mask.cpu().numpy(),
                        args.target
                    )

                    for i in range(len(decoded_results)):
                        if args.target == "material" or args.target == "uni_mat":
                            sentences, lattice, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Generated material:{sentences}")
                            dataset.data[index]["prediction"] = {
                                "lattice": lattice,
                                "coordinates": atom_coordinates,
                            }
                            fw.write(
                                json.dumps(dataset.data[index], default=convert_json_serializable) + "\n"
                            )
                            index += 1
                        elif args.target == "mol" or args.target == "uni_mol":
                            sentences, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Generated molecule: {sentences}")
                            dataset.data[index]["prediction"] = {
                                "coordinates": atom_coordinates,
                            }
                            fw.write(
                                json.dumps(dataset.data[index], default=convert_json_serializable) + "\n"
                            )
                            index += 1
                        elif args.target == "cond_mol":
                            sentences, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Conditional generated: {sentences}")
                            while index in skip_index_list:
                                index += 1
                            ans = dict()
                            ans["coordinates"] = atom_coordinates[1:]
                            ans["smi"] = "".join(smiles[i])
                            ans["prop"] = dataset.data[index]["prop"]
                            ans["prop_val"] = dataset.data[index]["prop_val"]
                            fw.write(json.dumps(ans, default=convert_json_serializable) + "\n")
                            index += 1
                    # endregion ---------------
        # endregion ---------------
    else:
        # region denovo case -------------
        #if inference_config.sample:
        num_batches = inference_config.sample_size // inference_config.infer_batch_size
        batches = []
        for _ in range(num_batches):
            input_ids = torch.full((inference_config.infer_batch_size, 1), tokenizer.bos_idx)
            coordinates_mask = torch.zeros((inference_config.infer_batch_size, inference_config.sample_max_length))
            batches.append({"input_ids": input_ids, "coordinates_mask": coordinates_mask})

        if inference_config.sample_size % inference_config.infer_batch_size != 0:
            remainder_size = inference_config.sample_size % inference_config.infer_batch_size
            input_ids = torch.full((remainder_size, 1), tokenizer.bos_idx)
            coordinates_mask = torch.zeros((remainder_size, inference_config.sample_max_length))
            batches.append({"input_ids": input_ids, "coordinates_mask": coordinates_mask})

        gen_config = GenerationConfig(
            pad_token_id=tokenizer.padding_idx,
            eos_token_id=tokenizer.coord_idx,
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )
        os.makedirs(inference_config.output_file, exist_ok=True)
        with open(os.path.join(inference_config.output_file, 'sample.txt'), "w") as fw:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(batches)):
                    batch = move_to_device(batch, "cuda")
                    ret = model.net.generate(
                        input_ids=batch["input_ids"],
                        coordinates_mask=batch["coordinates_mask"],
                        generation_config=gen_config,
                        max_length=batch["coordinates_mask"].shape[1],
                        do_sample=True,
                        top_p = inference_config.top_p,
                        temperature = inference_config.temperature,
                    )
                    tokens = ret.sequences.cpu().numpy()
                    ret = []
                    bs = tokens.shape[0]
                    for i in range(bs):
                        sent = []
                        if tokenizer.coord_idx in tokens[i]:
                            for j in range(len(tokens[i])):
                                if tokens[i][j] not in [tokenizer.bos_idx, tokenizer.eos_idx, tokenizer.padding_idx]:
                                    sent.append(tokenizer.get_tok(tokens[i][j]))
                                if tokens[i][j] == tokenizer.coord_idx:
                                    break
                            ret.append(sent)

                    for i in range(len(ret)):
                        sent = ret[i]
                        fw.write(' '.join(sent) + "\n")

                    tokens = [token[:np.where(token == tokenizer.coord_idx)[0][0] + 1].tolist() for token in tokens if tokenizer.coord_idx in token]
                    tokens = [token for token in tokens if len(token) <= 22]
                    origin_coordinates_mask = [[0 for _ in range(len(token))] + [1 for _ in range((len(token)+1))] + [0] for token in tokens]
                    max_tokens = max(len(token) for token in tokens)
                    max_masks = max(len(mask) for mask in origin_coordinates_mask)
                    input_ids = torch.cat(
                        [
                            pad_1d_unsqueeze(
                                torch.Tensor(token).long(),
                                max_tokens,
                                max_tokens - len(token),
                                tokenizer.padding_idx,
                            )
                            for token in tokens
                        ]
                    )
                    coordinates_mask = torch.cat(
                        [
                            pad_1d_unsqueeze(
                                torch.Tensor(mask).long(),
                                max_masks,
                                max_tokens - len(token),
                                tokenizer.padding_idx,
                            )
                            for mask, token in zip(origin_coordinates_mask, tokens)
                        ]
                    )

                    input_ids = move_to_device(input_ids, "cuda")
                    coordinates_mask = move_to_device(coordinates_mask, "cuda")
                    gen_config = GenerationConfig(
                        pad_token_id=tokenizer.padding_idx,
                        eos_token_id=tokenizer.eos_idx,
                        use_cache=True,
                        max_length=saved_config.max_position_embeddings,
                        return_dict_in_generate=True,
                    )
                    ret = model.net.generate(
                        input_ids=input_ids,
                        coordinates_mask=coordinates_mask,
                        do_sample=False,
                        generation_config=gen_config,
                        max_length=coordinates_mask.shape[1],
                    )
                    sentences = ret.sequences.cpu().numpy()
                    coordinates = ret.coordinates.cpu().numpy()
                    masks = coordinates_mask.cpu().numpy()
                    ret2 = tokenizer.decode_batch(sentences, coordinates, masks)

                    for i in range(len(ret2)):
                        sent, lattice, atom_coordinates = ret2[i]
                        if inference_config.verbose:
                            print(sent)
                        species = sent.split('<coord>')[0].split()
                        print(species)
                        if len(atom_coordinates) > len(species):
                            atom_coordinates = atom_coordinates[: len(species)]
                        try:
                            structure = Structure(
                                lattice=lattice,
                                species=species,
                                coords=atom_coordinates,
                            )
                            cif = CifWriter(structure)
                            cif.write_file(
                                f"{inference_config.output_file}/gen_{batch_idx * inference_config.infer_batch_size + i}.cif"
                            )
                        except:
                            print('fail')
            # endregion ---------------

if __name__ == "__main__":
    main()
