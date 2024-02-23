import torch
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import pathlib
from PIL import Image


class LabelCoder:
    def __init__(self, directory_path="./data") -> None:
        self.labels = [label for label in os.listdir(directory_path)]

    def get_index(self, value):
        return self.labels.index(value)

    def encode(self, t: list) -> torch.Tensor:
        for i, value in enumerate(t):
            t[i] = self.get_index(value)
        return torch.tensor(t, dtype=torch.uint8)

    def decode(self, t: torch.Tensor) -> list:
        temp = []
        for value in t:
            temp.append(value)
        for i, value in enumerate(t):
            temp[i] = self.labels[value]
        return temp


class DigitsDataset(Dataset):
    def __init__(self, directory_path="./data", transformer=None) -> None:
        super().__init__()
        self.paths_to_image = list(
            pathlib.Path(directory_path).glob("*/*.jpg"))
        # get all paths of images in directory_path
        self.transform = transformer
        self.classes = [i for i in os.listdir(directory_path)]

    def __len__(self):
        return len(self.paths_to_image)

    def __getitem__(self, index: int):
        image = Image.open(self.paths_to_image[index])
        label = self.paths_to_image[index].parent.name
        if self.transform:
            return self.transform(image), label
        else:
            return image, label


if __name__ == "__main__":
    transformer = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = DigitsDataset("./data", transformer=transformer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    image, label = next(iter(dataloader))
    image = torch.Tensor(image)
    print(image)
