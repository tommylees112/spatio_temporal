from pathlib import Path
from ruamel.yaml import YAML
from typing import Dict, Any, Union, List, Optional
from collections import OrderedDict
import pandas as pd
import pprint
import torch


class Config:
    _mandatory_keys: List[str] = [
        "batch_size",
        "data_dir",
        "target_variable",
        "n_epochs",
        "hidden_size",
        "loss",
        "optimizer",
        "horizon",
        "seq_length",
        "test_end_date",
        "test_start_date",
        "train_end_date",
        "train_start_date",
        "validation_end_date",
        "validation_start_date",
        "data_dir",
        "experiment_name",
    ]

    _defaults: Dict[str, Any] = {
        "autoregressive": False,
        "pixel_dims": ["lat", "lon"],
        "num_workers": 1,
        "seed": 1234,
        "device": "cuda",
        "learning_rate": 1e-2,
        "time_str": "time",
        "run_dir": Path("runs"),
        "forecast_variables": None,
        "static_inputs": None,
        "clip_gradient_norm": None,
        "validate_every_n": 1,
        "scheduler": None,
        "model": "lstm",
        "dropout": 0.4,
        "constant_mean": None,
        "constant_std": None,
        "early_stopping": None,
        "encode_doys": False,
        "input_variables": None,
        "static_data_path": None,
        "data_path": None,
        "normalize_variables": None,
    }

    def __init__(self, cfg_path: Path):
        self.file_path = cfg_path
        self._cfg = self._read_config(self.file_path)
        self._check_all_mandatory()
        self._check_training_data_paths_exist()

    def __repr__(self):
        return pprint.pformat(self._cfg)

    def _read_config(self, cfg_path: Path):
        if cfg_path.exists():
            with cfg_path.open("r") as fp:
                yaml = YAML(typ="safe")
                cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(cfg_path)

        cfg = self._parse_config(cfg)

        return cfg

    def _check_all_mandatory(self):
        for key in self._mandatory_keys:
            self.get_mandatory_attrs(key)

    def _inverse_transform_cfg_types(self) -> Dict[Any, Union[str, List[str]]]:
        """Convert the self._cfg back to strings for dumping to .yml file

        Returns:
            Dict[str, str]: Config file with str, str 
        """
        temp_cfg: Dict[Any, Union[str, List[str]]] = {}
        for key, val in self._cfg.items():
            #  convert Path objects to str
            if any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(str(elem))
                    temp_cfg[key] = temp_list
                else:
                    temp_cfg[key] = str(val)

            # convert pd.Timsestamp objects to str
            elif key.endswith("_date"):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(elem.strftime(format="%d/%m/%Y"))
                    temp_cfg[key] = temp_list
                else:
                    # Ignore None's due to e.g. using a per_basin_period_file
                    if isinstance(val, pd.Timestamp):
                        temp_cfg[key] = val.strftime(format="%d/%m/%Y")
            else:
                temp_cfg[key] = val
        return temp_cfg

    def _write_all_non_supplied_defaults_to_dict(self) -> None:
        # ensure that defaults are in the keys
        for key in self._defaults.keys():
            if not key in [l for l in self._cfg.keys()]:
                # write the default to self._cfg
                self.get_property_with_defaults(key)

    def dump_config(self, folder: Path, filename: str = "config.yml"):
        cfg_path = folder / filename
        if not cfg_path.exists():
            self._write_all_non_supplied_defaults_to_dict()
            with cfg_path.open("w") as fp:
                temp_cfg = self._inverse_transform_cfg_types()
                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            FileExistsError(cfg_path)

    def get_mandatory_attrs(self, key: str):
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key]

    def _get_device_property(self, key: str = "device") -> str:
        #  check (and set) defaults
        value = self.get_property_with_defaults(key)

        #  only use cuda if available
        #  catches errors that are difficult for users
        #  AssertionError: Torch not compiled with CUDA enabled
        if "cuda" in value:
            value = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._cfg[key] = value

        return self._cfg[key]

    def get_property_with_defaults(self, key: str) -> Any:
        # update the _cfg if access the default of an attribute
        if key not in self._cfg.keys():
            value = self._defaults[key]
            self._cfg[key] = value

        return self._cfg[key]

    def _check_training_data_paths_exist(self):
        # Optionally Provided! 
        if self.data_path is not None:
            assert self.data_path.exists(), f"Data path does not exist: {self.data_path}"
        if (self.static_inputs is not None) and (self.static_inputs != "embedding") and (self.static_data_path is not None):
            assert self.static_data_path.exists(), f"Static Data Path must be provided with variables: [{self.static_inputs}]"

    #  --------------------------------------------------
    #  - Parse Config -----------------------------------
    #  --------------------------------------------------
    def _parse_config(self, cfg: Dict[str, Any]):
        for key, val in cfg.items():
            #  convert paths to Path objects
            self._parse_path_objects(cfg, key, val)

            # convert dates to pd.Timestamp objects
            self._parse_date_objects(cfg, key, val)
        return cfg

    @staticmethod
    def _parse_path_objects(cfg: Dict[str, Any], key: str, val: Any) -> Dict[str, Any]:
        if any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
            if (val is not None) and (val != "None"):
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element).absolute())
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val).absolute()
            else:
                cfg[key] = None
        return cfg

    @staticmethod
    def _parse_date_objects(cfg: Dict[str, Any], key: str, val: Any) -> Dict[str, Any]:
        if key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")
        return cfg

    @staticmethod
    def _read_list_of_dicts_into_one_dict(
        read_list: Optional[List[Dict[str, float]]]
    ) -> Optional[Dict[str, Any]]:
        if read_list is not None:
            assert False, "self.constant_mean self.constant_std don't currently work!"
            return_dict = read_list[0]
            for dict_ in read_list[1:]:
                key = list(dict_.keys())[0]
                value = list(dict_.values())[0]
                return_dict[key] = value
        else:
            return_dict = None
        return return_dict

    #  --------------------------------------------------
    #  - Mandatory Properties ---------------------------
    #  --------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.get_mandatory_attrs("data_dir")

    @property
    def experiment_name(self) -> str:
        return self.get_mandatory_attrs("experiment_name")

    @property
    def batch_size(self) -> int:
        return self.get_mandatory_attrs("batch_size")

    @property
    def n_epochs(self) -> int:
        return self.get_mandatory_attrs("n_epochs")

    @property
    def hidden_size(self) -> Union[int, Dict[str, int]]:
        return self.get_mandatory_attrs("hidden_size")

    @property
    def loss(self) -> str:
        return self.get_mandatory_attrs("loss")

    @property
    def optimizer(self) -> str:
        return self.get_mandatory_attrs("optimizer")

    @property
    def horizon(self) -> int:
        return self.get_mandatory_attrs("horizon")

    @property
    def seq_length(self) -> int:
        return self.get_mandatory_attrs("seq_length")

    @property
    def target_variable(self) -> str:
        return self.get_mandatory_attrs("target_variable")

    @property
    def test_end_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("test_end_date")

    @property
    def test_start_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("test_start_date")

    @property
    def train_end_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("train_end_date")

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("train_start_date")

    @property
    def validation_end_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("validation_end_date")

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return self.get_mandatory_attrs("validation_start_date")

    #  --------------------------------------------------
    #  - Settable Properties ----------------------------
    #  --------------------------------------------------
    #  Run Dir
    @property
    def run_dir(self) -> Path:
        return self.get_property_with_defaults("run_dir")

    @run_dir.setter
    def run_dir(self, folder: Path):
        self._cfg["run_dir"] = folder

    #  Seed
    @property
    def seed(self) -> int:
        return self.get_property_with_defaults("seed")

    @seed.setter
    def seed(self, seed: int):
        self._cfg["seed"] = seed

    #  --------------------------------------------------
    #  - Properties with defaults -----------------------
    #  --------------------------------------------------
    @property
    def input_variables(self) -> List[str]:
        return self.get_property_with_defaults("input_variables")

    @property
    def autoregressive(self) -> bool:
        return self.get_property_with_defaults("autoregressive")

    @property
    def pixel_dims(self) -> List[str]:
        return self.get_property_with_defaults("pixel_dims")

    @property
    def num_workers(self) -> int:
        return self.get_property_with_defaults("num_workers")

    @property
    def device(self) -> str:
        return self._get_device_property("device")

    @property
    def learning_rate(self) -> float:
        return self.get_property_with_defaults("learning_rate")

    @property
    def time_str(self) -> str:
        return self.get_property_with_defaults("time_str")

    @property
    def forecast_variables(self) -> Optional[List[str]]:
        return self.get_property_with_defaults("forecast_variables")

    @property
    def static_inputs(self) -> Optional[List[str]]:
        return self.get_property_with_defaults("static_inputs")

    @property
    def clip_gradient_norm(self) -> float:
        return self.get_property_with_defaults("clip_gradient_norm")

    @property
    def validate_every_n(self) -> int:
        return self.get_property_with_defaults("validate_every_n")

    @property
    def scheduler(self) -> Optional[str]:
        return self.get_property_with_defaults("scheduler")

    @property
    def model(self) -> str:
        return self.get_property_with_defaults("model")

    @property
    def dropout(self) -> str:
        return self.get_property_with_defaults("dropout")

    @property
    def constant_mean(self) -> Optional[Dict[str, float]]:
        read_list: Optional[List[Dict[str, float]]] = self.get_property_with_defaults(
            "constant_mean"
        )
        #  TODO: better way of getting ruamel.Yaml to read a dict
        return_dict = self._read_list_of_dicts_into_one_dict(read_list)
        return return_dict

    @property
    def constant_std(self) -> Optional[Dict[str, float]]:
        read_list: Optional[List[Dict[str, float]]] = self.get_property_with_defaults(
            "constant_std"
        )
        #  TODO: better way of getting ruamel.Yaml to read a dict
        return_dict = self._read_list_of_dicts_into_one_dict(read_list)
        return return_dict

    @property
    def early_stopping(self) -> Optional[int]:
        return self.get_property_with_defaults("early_stopping")

    @property
    def encode_doys(self) -> bool:
        return self.get_property_with_defaults("encode_doys")

    #  Data Paths: Optional to allow user to pass in datasets directly
    @property
    def data_path(self) -> Path:
        return self.get_property_with_defaults("data_path")

    @property
    def static_data_path(self) -> Optional[Path]:
        return self.get_property_with_defaults("static_data_path")

    @property
    def normalize_variables(self) -> Optional[List[str]]:
        return self.get_property_with_defaults("normalize_variables")

