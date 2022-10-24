import hydra
import logging
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="defaults")
def main(config: DictConfig) -> None:
    pl.seed_everything(2509)
    logger.info("\n" + OmegaConf.to_yaml(config))

    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(config.model)
    data_module = hydra.utils.instantiate(config.data)

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='loss/val'), pl.callbacks.EarlyStopping(monitor='loss/val', patience=100)]

    trainer = pl.Trainer(**OmegaConf.to_container(config.trainer), logger=tensorboard, callbacks=callbacks)

    trainer.fit(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)  # Optional

if __name__ == '__main__':
    main()
