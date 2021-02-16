from pathlib import Path
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.config import Config


class TestTrainer:
    def test_trainer(self, tmp_path: Path):
        cfg = Config(Path("tests/testconfigs/config.yml"))
        cfg.run_dir = tmp_path
        Trainer(cfg=cfg)
        assert False, "Check where the run_dir is created"


class TestTester:
    def test_tester(self):
        cfg = Config(Path("tests/testconfigs/config.yml"))
        Tester(cfg=cfg)
