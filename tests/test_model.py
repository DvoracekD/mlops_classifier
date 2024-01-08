import torch
from classifier.models.model import MyNeuralNet

def test_model():
    batch_n = 64
    input = torch.randn(batch_n, 28, 28)
    model = MyNeuralNet()
    output = model(input)

    assert output.shape == (batch_n, 10)
