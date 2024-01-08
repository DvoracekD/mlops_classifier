import torch
import pytest
from classifier.models.model import MyNeuralNet

def test_model():
    batch_n = 64
    input = torch.randn(batch_n, 1, 28, 28)
    model = MyNeuralNet()
    output = model(input)

    assert output.shape == (batch_n, 10)

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model = MyNeuralNet()
        model(torch.randn(1,2,3))