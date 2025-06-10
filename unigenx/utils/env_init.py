# -*- coding: utf-8 -*-
import io
import json
import os

import torch

from unigenx.logging import logger


def set_env(args):
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if os.environ.get("LOCAL_RANK") is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
        torch.cuda.set_device(args.local_rank)

        logger.success(
            "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
                os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
            )
        )
