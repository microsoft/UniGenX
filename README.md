# UniGenX

> **⚠️ This repository is no longer maintained.**
>
> Please use the actively maintained version instead:
> **https://github.com/UniGenX-Team/UniGenX**
>
> The maintained repository includes up-to-date code, example inputs, data
> preparation instructions, and generation scripts. This repository is kept only
> for historical reference and is expected to be archived.

UniGenX is a unified generative foundation model that jointly generates sequences
and 3D structures for molecules, materials, and proteins.

- **Paper:** *UniGenX: Unified Generation of Sequence and Structure with Numeric
  Diffusion*, arXiv [2503.06687](https://arxiv.org/abs/2503.06687)
- **Benchmarks / datasets:** QM9, GEOM-Drugs, MP-20 (plus Carbon-24, MPTS-52 and
  protein-ligand data). These are public datasets; see the maintained repository
  above for example inputs and preparation steps.

## Security notice

Loading datasets from `pickle`/`pkl` files and from LMDB databases has been
**removed** from this repository. Those formats relied on Python `pickle`
deserialization, which can execute arbitrary code when reading untrusted files
(CWE-502). Use JSON/JSONL inputs instead and pass a `.json` or `.jsonl` file via
`--input_file`. Example inputs are available in the maintained repository:
https://github.com/UniGenX-Team/UniGenX/tree/main/examples/data

Model checkpoints are now loaded with `torch.load(..., weights_only=True)` so that
reading a checkpoint cannot execute arbitrary pickle payloads.

## Installation

First create a conda environment from `environment.yaml`, then run
`pip install -r requirements.txt`.

> ATTENTION: Please ensure that the torch family build you install matches your
> CUDA version.
