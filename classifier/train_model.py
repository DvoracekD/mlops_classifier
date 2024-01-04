import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from classifier.models.model import MyNeuralNet

lr = 1e-3
batch_size = 64

if __name__ == "__main__":
    """Train a model on MNIST."""

    # TODO: Implement training loop here
    model = MyNeuralNet()
    train_data = torch.load("data/processed/train_data.pt")
    train_labels = torch.load("data/processed/train_labels.pt")
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 1
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(loss.item())

        else:
            loss = running_loss / len(train_loader)
            test_losses.append(loss)
            print(f"Training loss: {loss}")

    plt.plot(train_losses)
    plt.savefig("reports/figures/training.png")
    torch.save(model.state_dict(), "models/trained_model.pt")
