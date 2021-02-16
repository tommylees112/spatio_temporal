from pathlib import Path
from spatio_temporal.config import Config
import pandas as pd
import numpy as np
from ruamel.yaml import YAML


class TestConfig:
    def test_config(self):
        paths = list(Path("tests/testconfigs").glob("*.yml"))
        for path in paths:
            cfg = Config(cfg_path=path)

            assert isinstance(cfg.test_start_date, pd.Timestamp)
            assert isinstance(cfg.data_dir, Path)

            assert all(np.isin(cfg._mandatory_keys, list(dir(cfg))))
            assert all(np.isin(cfg._mandatory_keys, list(cfg._cfg.keys())))
            assert all(np.isin(list(cfg._defaults.keys()), list(dir(cfg))))

            if cfg.file_path.name == "test_1d_config.yml":
                assert cfg.pixel_dims == ["pixel"]

            # TODO: test default args
            original = cfg._cfg.pop("pixel_dims", None)
            assert cfg.pixel_dims == ["lat", "lon"], "Expect to return the default"

    def test_dump_config(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir(exist_ok=True, parents=True)
        path = Path("tests/testconfigs/test_config.yml")

        cfg = Config(cfg_path=path)
        cfg.run_dir = run_dir

        #  check that defaults not specified are written to file
        # check that the file gets created
        cfg.dump_config(run_dir)

        assert "config.yml" in [l.name for l in run_dir.glob("*")]

        cfg_path = run_dir / "config.yml"
        with cfg_path.open("r") as fp:
            yaml = YAML(typ="safe")
            cfg2 = yaml.load(fp)

        expected_keys_with_defaults = [
            "autoregressive",
            "pixel_dims",
            "num_workers",
            "seed",
            "device",
            "learning_rate",
            "time_str",
            "run_dir",
        ]
        for key in expected_keys_with_defaults:
            assert key in [l for l in cfg2.keys()]
