import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CheXpertDataset(Dataset):
    """CheXpert dataset loader"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        self.frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # we need to rescale for the same size as vgg16 expects
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # we need to concert to RGB because chestxpert is grayscal but vgg16 expects 3 channels
        image = Image.open(self.frame.iloc[idx, 0]).convert('RGB')
        label = self.frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
