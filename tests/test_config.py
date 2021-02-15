from pathlib import Path
from spatio_temporal.config import Config
import pandas as pd
import numpy as np


class TestConfig:
    def test_config(self):
        cfg = Config(cfg_path=Path("tests/testconfigs/config.yml"))

        assert isinstance(cfg.test_start_date, pd.Timestamp)
        assert isinstance(cfg.data_dir, Path)

        assert all(np.isin(cfg._mandatory_keys, list(dir(cfg))))
        assert all(np.isin(cfg._mandatory_keys, list(cfg._cfg.keys())))
        assert all(np.isin(list(cfg._defaults.keys()), list(dir(cfg))))

        assert False
