import torch


def normalize_tensor(tensor):
    mean, std = tensor.mean(), tensor.std()
    return (tensor - mean) / std


if __name__ == "__main__":
    train_data, train_labels = [], []
    for i in range(10):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    torch.save(normalize_tensor(train_data), "data/processed/train_data.pt")
    torch.save(train_labels, "data/processed/train_labels.pt")

    torch.save(normalize_tensor(test_data), "data/processed/test_data.pt")
    torch.save(test_labels, "data/processed/test_labels.pt")
