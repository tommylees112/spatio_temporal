from pathlib import Path
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.config import Config


class TestTrainer:
    def test_trainer(self):
        cfg = Config(Path("tests/testconfigs/config.yml"))
        Trainer(cfg=cfg)
        assert False, "Move data from "


class TestTester:
    def test_tester(self):
        cfg = Config(Path("tests/testconfigs/config.yml"))
        Tester(cfg=cfg)