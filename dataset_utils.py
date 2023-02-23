import os
import torchvision
from torchvision import transforms


def load_dataset(use_train: bool = False) -> torchvision.datasets.DatasetFolder:
    # # dataset downloaded from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
    dir_name: str = "val"
    if use_train:
        dir_name = "train"

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
