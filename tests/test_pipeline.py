from pathlib import Path
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import create_linear_ds
from spatio_temporal.training.eval_utils import save_loss_curves, save_timeseries


class TestLinear:
    def test_linear_example(self):
        ds = create_linear_ds(epsilon_sigma=10)
        cfg = Config(Path("tests/testconfigs/test_config.yml"))

        #  Train
        trainer = Trainer(cfg, ds)
        losses = trainer.train_and_validate()
        save_loss_curves(losses, cfg)

        # Test
        tester = Tester(cfg, ds)
        preds = tester.run_test()
        for _ in range(2):
            save_timeseries(preds, cfg)

    def test_vci_example(self):
        pass


if __name__ == "__main__":
    t = TestLinear()
    t.test_linear_example()
