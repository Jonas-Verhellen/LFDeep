import hydra
import logging
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
logger = logging.getLogger(__name__)
import torch
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="defaults")
def main(config: DictConfig) -> None:
    """
    
    Main function for the LFdeeP project: a Hydra-based PyTorch Lightning training script.

    Args:
        config (DictConfig): The configuration for your training process.

    This function sets up and runs a PyTorch Lightning training loop based on the provided configuration.
    It instantiates the model and data modules, manages directory outputs using TensorBoard, and logs important training information.

    Args:
        - config (DictConfig): The configuration for the training process, including model, data, and trainer settings.

    The main function follows these steps:
    1. Empties the CUDA cache for GPU memory management.
    2. Seeds the random number generator for reproducibility.
    3. Logs the configuration using OmegaConf.
    4. Instantiates the specified model and data modules.
    5. Sets up TensorBoard for logging with optional graph visualization.
    6. Configures model checkpointing and early stopping callbacks.
    7. Initializes the PyTorch Lightning Trainer with provided settings.
    8. Fits the model to the data.
    9. Optionally tests the trained model.

    Returns:
        None

    """
    torch.cuda.empty_cache()
    pl.seed_everything(2509)
    logger.info("\n" + OmegaConf.to_yaml(config))

    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(config.model)
    data_module = hydra.utils.instantiate(config.data)

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=2, monitor="loss/val", mode="min", filename="sample-mh-{epoch:02d}-{val_loss:.2f}",)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='loss/val', patience=500)
    callbacks = [checkpoint_callback, early_stopping_callback]

    trainer = pl.Trainer(**OmegaConf.to_container(config.trainer),  logger=tensorboard, callbacks=callbacks, auto_lr_find=True, precision=16, accelerator='gpu', devices=4, strategy="ddp")
    trainer.fit(model, datamodule=data_module) 
    trainer.test(model, datamodule=data_module)  # Optional


if __name__ == '__main__':
    main()
