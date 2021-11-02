from torch import Tensor
from typing import Dict, Union, Any, cast
import re
from pathlib import Path
from spatio_temporal.config import Config
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.model.bi_lstm import BiLSTM
import torch.nn as nn


def _to_device(
    data: Union[Dict[str, Tensor], Dict[str, Dict[str, Tensor]]], device: str
) -> Dict[str, Tensor]:
    for key in data.keys():
        if isinstance(data[key], dict):  # type: ignore
            for nested_key in data[key].keys():  # type: ignore
                data[key][nested_key].to(device)  # type: ignore
        else:
            data[key] = data[key].to(device)  # type: ignore

    return data  # type: ignore


def get_model(input_size: int, output_size: int, cfg: Config) -> nn.Module:
    #  TODO: def get_model from lookup: Dict[str, Model]
    model_str = cfg.model.lower()
    model: nn.Module

    if model_str == "lstm":
        model = LSTM(
            input_size=input_size,
            hidden_size=cast(int, cfg.hidden_size),  #  make mypy happy
            output_size=output_size,
            forecast_horizon=cfg.horizon,
            initial_forget_bias=cfg.initial_forget_bias,
            dropout_rate=cfg.dropout,
        ).to(cfg.device)
    elif model_str == "bilstm":
        model = BiLSTM(
            input_size=input_size,
            hidden_size=cast(int, cfg.hidden_size),  # make mypy happy
            output_size=output_size,
            forecast_horizon=cfg.horizon,
            initial_forget_bias=cfg.initial_forget_bias,
            dropout_rate=cfg.dropout,
        ).to(cfg.device)

    else:
        assert False, f"{model} is not a valid choice for model"

    return model


def has_datetime(path: Path) -> bool:
    search = path.name
    regex = r"\d{4}_\d{6}"
    return [l for l in re.finditer(regex, search)] != []
