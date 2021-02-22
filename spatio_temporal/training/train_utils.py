from torch import Tensor
from typing import Dict


def _to_device(data: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    for key in data.keys():
        if isinstance(data[key], dict):
            for nested_key in data[key].keys():
                data[key][nested_key].to(device)
        else:
            data[key] = data[key].to(device)

    return data
