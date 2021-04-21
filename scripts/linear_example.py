from pathlib import Path
from spatio_temporal.config import Config
from spatio_temporal.training.trainer import Trainer
from spatio_temporal.training.tester import Tester
from tests.utils import create_linear_ds
from spatio_temporal.training.eval_utils import save_loss_curves, save_timeseries
from tqdm import tqdm


if __name__ == "__main__":
    #  EXPLICITLY write out training loop (good for debugging)
    ds = create_linear_ds(epsilon_sigma=10)
    cfg = Config(Path("tests/testconfigs/test_config.yml"))

    #  Train
    trainer = Trainer(cfg, ds)
    tester = Tester(cfg, ds)

    cfg._cfg["n_epochs"] = 2
    trainer.train_and_validate()
    preds = tester.run_test()

    assert False

    ## Test one loop
    #  Items for training loop
    model = trainer.model
    optimizer = trainer.optimizer
    loss_fn = trainer.loss_fn
    dl = trainer.train_dl

    epoch = 0
    pbar = tqdm(dl, desc=f"Training Epoch {epoch}: ")

    # Single forward pass
    data = pbar.__iter__().__next__()

    optimizer.zero_grad()
    y_hat = model(data)
    y = data["y"]
    loss = loss_fn(y_hat["y_hat"], y)
