from pathlib import Path
from spatio_temporal.config import Config
import pandas as pd
import numpy as np


class TestConfig:
    def test_config(self):
        paths = [Path("tests/testconfigs/config.yml"), Path("tests/testconfigs/test_config.yml")]
        for path in paths:
            cfg = Config(cfg_path=path)

            assert isinstance(cfg.test_start_date, pd.Timestamp)
            assert isinstance(cfg.data_dir, Path)

            assert all(np.isin(cfg._mandatory_keys, list(dir(cfg))))
            assert all(np.isin(cfg._mandatory_keys, list(cfg._cfg.keys())))
            assert all(np.isin(list(cfg._defaults.keys()), list(dir(cfg))))

    def test_dump_config(self):
        # check that defaults not specified are written to file
        # check that the file gets created
        assert False