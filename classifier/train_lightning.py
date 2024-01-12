from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from classifier.data.dataset import train_dataset
from classifier.models.lightning_model import MyNeuralNet

if __name__ == "__main__":
    """Train a model on MNIST."""

    model = MyNeuralNet()
    trainer = Trainer(
        max_epochs=10,
        precision="bf16-true",
        limit_train_batches=0.2,
        profiler="simple",
        callbacks=[EarlyStopping(monitor="val_loss")],
        logger=WandbLogger(project="dtu_mlops"),
    )

    # Load the dataset and split it into training and validation sets
    dataset = train_dataset()
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, [train_len, val_len])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=64)

    train_loader = DataLoader(train_dataset(), batch_size=64)
    trainer.fit(model, train_loader, val_loader)
