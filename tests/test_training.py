import torch
from torch import nn, optim
from classifier.train_model import train_epoch

def test_train_epoch():
    model = nn.Linear(2, 2)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    data_point = torch.randn((1, 2)), torch.tensor(0).view(1)  # Assuming a 3-channel image with size 64x64
    tensor_dataset = torch.utils.data.TensorDataset(*data_point)
    train_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1, shuffle=True)

    train_epoch(0, train_loader, optimizer, model, criterion)

    assert model.weight.grad is not None, "gradient is not propagated after one epoch"
