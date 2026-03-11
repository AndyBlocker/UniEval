"""SNNWrapper, Judger, reset utilities, and attn_convert."""

import torch
import torch.nn as nn
from copy import deepcopy

from ..operators.neurons import IFNeuron, ORIIFNeuron
from ..operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
from ..operators.attention import SAttention
from ..quantization.lsq import QAttention
from .converter import SNNConverter
from .rules import DEFAULT_CONVERSION_RULES


def reset_model(model):
    """Recursively reset all SNN operator states in the model."""
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, (IFNeuron, LLConv2d, LLLinear, SAttention,
                              Spiking_LayerNorm, ORIIFNeuron)):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)


def attn_convert(QAttn, SAttn, level, neuron_type):
    """Transfer learned thresholds from QAttention to SAttention neurons.

    Maps quantization scale parameters to neuron firing thresholds.

    Args:
        QAttn: Source QAttention module.
        SAttn: Target SAttention module.
        level: Quantization level.
        neuron_type: Neuron type string.
    """
    SAttn.qkv = LLLinear(linear=QAttn.qkv, neuron_type="ST-BIF", level=level)
    SAttn.proj = LLLinear(linear=QAttn.proj, neuron_type="ST-BIF", level=level)

    # Transfer Q
    SAttn.q_IF.neuron_type = neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.is_init = False

    # Transfer K
    SAttn.k_IF.neuron_type = neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.is_init = False

    # Transfer V
    SAttn.v_IF.neuron_type = neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.is_init = False

    # Transfer attn
    SAttn.attn_IF.neuron_type = neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    # Transfer after_attn
    SAttn.after_attn_IF.neuron_type = neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False

    # Transfer proj
    SAttn.proj_IF.neuron_type = neuron_type
    SAttn.proj_IF.level = level
    SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    SAttn.proj_IF.is_init = False

    # Transfer dropout
    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop


class Judger:
    """Determines when SNN inference is complete by checking is_work flags.

    Tracks whether all neurons and layers have stabilized (is_work == False).
    """

    def __init__(self):
        self.network_finish = True

    def judge_finish(self, model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, (IFNeuron, LLLinear, LLConv2d)):
                self.network_finish = (
                    self.network_finish and (not model._modules[name].is_work)
                )
                is_need = True
            if not is_need:
                self.judge_finish(child)

    def reset_network_finish_flag(self):
        self.network_finish = True


def get_subtensors(tensor, sample_grain=255):
    """Rate coding: divide input across timesteps."""
    pieces = []
    for _ in range(int(sample_grain)):
        pieces.append(tensor / sample_grain)
    return torch.stack(pieces, dim=0)


class SNNWrapper(nn.Module):
    """Wraps a quantized ANN model for temporal SNN inference.

    Handles:
    - ANN-to-SNN conversion via SNNConverter
    - Temporal inference loop with rate/analog encoding
    - Early termination via Judger
    - ViT-specific pos_embed/cls_token handling

    Args:
        ann_model: The quantized ANN model to convert.
        time_step: Maximum number of timesteps.
        encoding_type: "rate" or "analog".
        level: Quantization level.
        neuron_type: Neuron type string.
        model_name: Model name (used for ViT detection).
        is_softmax: Whether attention uses softmax.
        converter: Optional SNNConverter instance.
    """

    def __init__(
        self,
        ann_model,
        time_step=2000,
        encoding_type="rate",
        level=16,
        neuron_type="ST-BIF",
        model_name="vit",
        is_softmax=True,
        converter=None,
    ):
        super().__init__()
        self.T = time_step
        self.finish_judger = Judger()
        self.Encoding_type = encoding_type
        self.level = level
        self.neuron_type = neuron_type
        self.model = ann_model
        self.model_name = model_name
        self.is_softmax = is_softmax
        self.max_T = 0

        # Save ViT embeddings before conversion
        if "vit" in self.model_name:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        # Convert using rule-based converter
        conv = converter or SNNConverter()
        conv.convert(
            self.model,
            level=self.level,
            neuron_type=self.neuron_type,
            is_softmax=self.is_softmax,
        )

    def reset(self):
        """Reset model state for new sample."""
        if "vit" in self.model_name:
            self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        reset_model(self)

    def forward(self, x, verbose=False):
        accu = None
        count1 = 0
        accu_per_timestep = []

        if self.Encoding_type == "rate":
            x = get_subtensors(x, sample_grain=self.level)

        while True:
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish

            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break

            # Zero pos_embed/cls_token after first timestep for ViT
            if "vit" in self.model_name and count1 > 0:
                self.model.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        self.model.patch_embed.num_patches + 1,
                        self.model.embed_dim,
                    ).to(x.device if torch.is_tensor(x) else "cuda")
                )
                self.model.cls_token = nn.Parameter(
                    torch.zeros(1, 1, self.model.embed_dim).to(
                        x.device if torch.is_tensor(x) else "cuda"
                    )
                )

            # Prepare input for this timestep
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input_t = x[count1]
                else:
                    input_t = torch.zeros(x[0].shape).to(x.device)
            else:  # analog
                if count1 == 0:
                    input_t = x
                else:
                    input_t = torch.zeros(x.shape).to(x.device)

            output = self.model(input_t)

            if count1 == 0:
                accu = output + 0.0
            else:
                accu = accu + output

            if verbose:
                accu_per_timestep.append(accu.clone())

            count1 += 1
            if count1 % 100 == 0:
                print(count1)

        print(f"\nTime Step: {count1}")

        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep, dim=0)
            return accu, count1, accu_per_timestep
        else:
            return accu, count1
