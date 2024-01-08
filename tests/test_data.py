import os.path
import pytest

import classifier.data.dataset as data
from tests import _PATH_DATA

print(_PATH_DATA)

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    dataset = data.train_dataset()
    assert len(dataset) == 50000
    img, label =  next(iter(dataset))
    assert img.shape == (1, 28, 28)

    unique_labels = set(range(10))
    all_labels_present = all(label in unique_labels for label in set(dataset.tensors[1].numpy()))
    assert all_labels_present, "Not all labels are represented in the dataset."
    