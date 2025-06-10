# -*- coding: utf-8 -*-
from torch import nn
from dataclasses import dataclass
from typing import Optional,Dict
from unigenx.logging import logger
import torch

@dataclass
class ModelOutput:
    loss: torch.Tensor
    num_examples: Optional[int] = None
    log_output: Optional[Dict] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None

class CrystalCriterions(nn.Module):
    def __init__(self, vocab_size, reduction="mean") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.word_loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        # self.sg_loss = nn.CrossEntropyLoss(reduction=reduction)
        # self.pos_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        model_output,
        batch_data,
    ):
        word_logits = model_output.logits
        loss_coord = model_output.loss
        bs, seqlen = word_logits.shape[:2]
        # y_0 = model_output.coordinates
        # note that y_0 has already been shifted and in the desired shape
        # shift so that tokens < n predict n
        label_ids = batch_data["label_ids"]
        # label_coordinates = batch_data["label_coordinates"]
        coordinates_mask = batch_data["coordinates_mask"]

        shift_label_ids = label_ids[..., 1:].contiguous()
        shift_label_coordinates_mask = coordinates_mask[..., 1:].contiguous()
        shift_word_logits_ = word_logits[:, :-1, :].contiguous()

        shift_word_logits = shift_word_logits_[~shift_label_coordinates_mask.bool()]

        # Calculate loss on word tokens
        shift_words_labels = shift_label_ids[~shift_label_coordinates_mask.bool()]

        loss_words = self.word_loss(
            shift_word_logits.view(-1, self.vocab_size),
            shift_words_labels.view(-1),
        )

        # Calculate loss on coordinate tokens
        # if label_coordinates.dtype != y_0.dtype:
        #     label_coordinates = label_coordinates.to(y_0.dtype)
        # loss_coord = self.pos_loss(y_0, label_coordinates)
        # Combine losses
        loss = loss_words + 100 * loss_coord
        loss_log = {
            "loss": loss.item() if loss is not None else None,
            "loss_words": loss_words.item() if loss_words is not None else None,
            "loss_y_0": loss_coord.item() if loss_coord is not None else None,
        }

        model_output.loss = loss
        model_output.num_examples = bs
        model_output.log_output = loss_log
        return model_output

