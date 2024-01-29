import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from chexpert_dataset import CheXpertDataset


# main
def get_mean_std(filename):

    # Define a transform to convert images to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load your dataset
    dataset = CheXpertDataset(filename, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std
