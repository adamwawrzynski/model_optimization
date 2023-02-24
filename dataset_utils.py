import os
import torch
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
from more_itertools import chunked


def load_dataset_cv() -> torch.utils.data.Dataset:
    # dataset downloaded from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
    dir_name: str = "val"
    data_dir: str = "./data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    testing_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "imagenet-mini", dir_name),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    return testing_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_dataset_nlp() -> torch.utils.data.Dataset:
    model_name: str = "textattack/bert-base-uncased-imdb"
    dataset_size: int = 15000
    max_length: int = 500
    batch_size: int = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(path="imdb")

    x_data = []
    y_data = []
    X = list(chunked([d["text"] for index, d in enumerate(dataset["test"]) if index < dataset_size], batch_size))
    Y = list(chunked([d["label"] for index, d in enumerate(dataset["test"]) if index < dataset_size], batch_size))

    for x_batch, y_batch in zip(X, Y):
        batch_encoding_sample = tokenizer(x_batch, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        x_data.append(batch_encoding_sample)
        y_data.append(torch.tensor(y_batch))

    return CustomDataset(x=x_data, y=y_data)
