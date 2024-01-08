import torch

def train_dataset():
    train_data = torch.load("data/processed/train_data.pt")
    train_labels = torch.load("data/processed/train_labels.pt")
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    return train_dataset
