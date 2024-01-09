import click
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from classifier.models.model import MyNeuralNet


@click.command()
@click.argument("model")
@click.argument("data")
def main(model, data):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyNeuralNet()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    with torch.no_grad():
        model.eval()
        acc = []
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            acc.append(accuracy.item())
        print(f"Accuracy: {torch.tensor(acc).mean()*100}%")
    model.train()


def predict():
    pass


if __name__ == "__main__":
    main()
