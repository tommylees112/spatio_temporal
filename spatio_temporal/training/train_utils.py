from torch import Tensor
from typing import Dict
from typing import Any
from spatio_temporal.config import Config
from spatio_temporal.model.lstm import LSTM
from spatio_temporal.model.bi_lstm import BiLSTM


def _to_device(data: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    for key in data.keys():
        if isinstance(data[key], dict):
            for nested_key in data[key].keys():
                data[key][nested_key].to(device)
        else:
            data[key] = data[key].to(device)

    return data


def get_model(
    input_size: int,
    output_size: int,
    cfg: Config
) -> Any:
    #  TODO: def get_model from lookup: Dict[str, Model]
    model_str = cfg.model.lower()
    if model_str == "lstm":
        model = LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            output_size=output_size,
            forecast_horizon=cfg.horizon,
        ).to(cfg.device)
    elif model_str == "bilstm":
        model = BiLSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            output_size=output_size,
            forecast_horizon=cfg.horizon,
        ).to(cfg.device)

    else:
        assert False, f"{model} is not a valid choice for model"

    return model
