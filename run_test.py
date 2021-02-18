from torch.utils.data import DataLoader

#
from tests.dummy_dataset import DummyDataset
from spatio_temporal.model.linear_regression import LinearRegression


if __name__ == "__main__":
    #  Get the config file

    #  RUN DUMMY EXPERIMENT
    dummy_dataset = DummyDataset()
    dl = train_dl = DataLoader(
        dummy_dataset, batch_size=10, shuffle=True, num_workers=0
    )

    #  create the model
    model = LinearRegression(
        input_size=dl.input_size * cfg.seq_length,
        output_size=dl.output_size,
        forecast_horizon=cfg.horizon,
    ).to(cfg.device)
