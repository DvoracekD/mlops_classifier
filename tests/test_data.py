import classifier.data.dataset as data

def test_data():
    dataset = data.train_dataset()
    assert len(dataset) == 50000
    img, label =  next(iter(dataset))
    assert img.shape == (1, 28, 28)

    unique_labels = set(range(10))
    all_labels_present = all(label in unique_labels for label in set(dataset.tensors[1].numpy()))
    assert all_labels_present, "Not all labels are represented in the dataset."
    