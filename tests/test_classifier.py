from pytorch_lightning import Trainer, seed_everything
from project.model.classifier import LitClassifier
from project.data import MNISTDataModule


def test_lit_classifier():
    seed_everything(1234)

    model = LitClassifier()
    dm = MNISTDataModule()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, dm)

    results = trainer.test(datamodule=dm)
    assert results[0]['test_acc'] > 0.7
