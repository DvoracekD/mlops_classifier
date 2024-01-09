import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn, optim

import classifier.data.dataset as data
from classifier.models.model import MyNeuralNet

lr = 1e-3
batch_size = 64


def train_epoch(epoch, train_loader, optimizer, model, criterion, train_losses=[], test_losses=[]):
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


def train(train_dataset, model):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 1
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        train_epoch(e, train_loader, optimizer, model, criterion, train_losses, test_losses)

    plt.plot(train_losses)
    plt.savefig("reports/figures/training.png")
    torch.save(model.state_dict(), "models/trained_model.pt")


def asses_model(train_dataset, model):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    preds, target = [], []
    for batch in train_loader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("reports/classification_report.txt", "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confmat)
    disp.plot()
    plt.savefig("reports/confusion_matrix.png")


if __name__ == "__main__":
    model = MyNeuralNet()
    train_dataset = data.train_dataset()

    train(train_dataset, model)
    asses_model(train_dataset, model)
