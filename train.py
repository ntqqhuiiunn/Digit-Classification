import torch, torch.nn as nn
from model import DigitsModel
from dataset import DigitsDataset, LabelCoder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device 

def label_process(label):
    labelEncoder = LabelCoder("./data")
    out_label = labelEncoder.encode(list(label))
    return out_label

def train(model : nn.Module, dataloader : DataLoader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params= model.parameters(), lr= 0.001)
    total_step = len(dataloader)
    num_epochs = 100
    loss_values = []
    for epoch in range(num_epochs):
        for i, (image, label)  in enumerate(dataloader):
            image = torch.Tensor(image)
            label = label_process(label)
            image = image.to(select_device())
            label = label.to(select_device())

            output = model(image)
            loss_value = loss_function(output, label)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss_value.item()))
            loss_values.append(loss_value.item())
    torch.save(model.state_dict(), './result/weights.ckpt')
    print("Training process end!")
    return loss_values


if __name__ == "__main__":
    transformer = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = DigitsDataset("./data", transformer= transformer)
    dataloader = DataLoader(dataset= dataset, batch_size= 16, num_workers= 1, shuffle= True)
    model = DigitsModel()
    losses = train(model, dataloader)
    # Load model using: model.load_state_dict(torch.load('weigths.ckpt'))
    plt.plot(losses)
    plt.show()

    

    