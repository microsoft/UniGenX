# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import unigenx.model.unigenx as model
from unigenx.logging import logger
from unigenx.model.modules.criterions import CrystalCriterions,ModelOutput


class UniGenX(nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, config, not_init=False):
        super().__init__()
        if not_init:
            return
        
        self.loss = CrystalCriterions(config.vocab_size)

        self.config = config
        self.net = model.UniGenX(config)

    def forward(self,batched_data, **kwargs):
        return self.net(**batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        return self.loss(model_output, batch_data)

    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

    
