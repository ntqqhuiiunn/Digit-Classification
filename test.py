import torch
from model import DigitsModel
from dataset import LabelCoder, DigitsDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def accurate(output: torch.Tensor, label: list):
    _, predicted = torch.max(output.data, 1)
    labelDecoder = LabelCoder("./data")
    predicted_result = labelDecoder.decode(predicted)
    batch_size = len(label)
    count = 0
    for i in range(batch_size):
        if predicted_result[i] == label[i]:
            count += 1
    return float(count / batch_size)


def test(model: torch.nn.Module, dataloader: DataLoader):
    num_epochs = 100
    accuracies = []
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(dataloader):
            image = torch.Tensor(image)
            output = model(image)
            accuracy = accurate(output, label)
            print(accuracy)
            accuracies.append(accuracy)
    return accuracies


if __name__ == "__main__":
    model = DigitsModel()
    model.load_state_dict(torch.load('./result/weights.ckpt'))

    transformer = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = DigitsDataset("./data", transformer=transformer)
    dataloader = DataLoader(dataset=dataset, batch_size=8,
                            num_workers=1, shuffle=True)

    image, label = next(iter(dataloader))
    image = torch.Tensor(image)
    outpt = model(image)
