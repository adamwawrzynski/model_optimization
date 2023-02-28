import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torchvision
from datasets import load_dataset
from more_itertools import chunked
from torchvision import transforms
from transformers import AutoTokenizer, BatchEncoding, GPT2Tokenizer, GPT2TokenizerFast


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[BatchEncoding], labels: List[torch.Tensor]):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class DatasetFactory(ABC):
    @abstractmethod
    def get_dataset(self) -> torch.utils.data.Dataset:
        ...

    def get_example_inputs(self) -> Optional[List[torch.Tensor]]:
        sample = self.get_dataset()[0][0]
        example_inputs: Optional[List[torch.Tensor]] = None
        if isinstance(sample, BatchEncoding):
            example_inputs = list(sample.values())

        return example_inputs


class DatasetImagenetMiniFactory(DatasetFactory):
    def __init__(
        self,
        data_dir: str,
        subset_name: str,
    ):
        self.data_dir = data_dir
        self.subset_name = subset_name

    def get_dataset(self) -> torch.utils.data.Dataset:
        # dataset downloaded from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        testing_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "imagenet-mini", self.subset_name),
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


class DatasetIMDBFactory(DatasetFactory):
    def __init__(
        self,
        pretrained_model_name: str,
        dataset_size: int,
        max_length: int,
        batch_size: int,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.dataset_size = dataset_size
        self.max_length = max_length
        self.batch_size = batch_size

    def get_dataset(self) -> torch.utils.data.Dataset:
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, model_max_length=self.max_length
        )

        # PAD_TOKEN is not set by default; set PAD_TOKEN for GPT model
        if isinstance(tokenizer, GPT2Tokenizer) or isinstance(
            tokenizer, GPT2TokenizerFast
        ):
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(path="imdb")

        samples: List[BatchEncoding] = []
        labels: List[torch.Tensor] = []
        sample_batches = list(
            chunked(
                [
                    d["text"]
                    for index, d in enumerate(dataset["test"])
                    if index < self.dataset_size
                ],
                self.batch_size,
            )
        )
        label_batches = list(
            chunked(
                [
                    d["label"]
                    for index, d in enumerate(dataset["test"])
                    if index < self.dataset_size
                ],
                self.batch_size,
            )
        )

        for x_batch, y_batch in zip(sample_batches, label_batches):
            batch_encoding_sample = tokenizer(
                x_batch,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )
            samples.append(batch_encoding_sample)
            labels.append(torch.tensor(y_batch))

        return CustomDataset(data=samples, labels=labels)
