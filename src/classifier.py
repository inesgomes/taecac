import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch import save, load
import wandb
import pandas as pd
from dataset import CheXpertDataset
from torch.utils.data import DataLoader
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# TODO hyperparameter tuning

DEVICE="mps"


def train_model(n_epochs, n_labels, train_loader):

    # load pretrained vgg16
    mdl = vgg16(pretrained=True)
    # substitute last layer from classifier
    mdl.classifier[6] = nn.Linear(4096, n_labels)
    # freeze parameters
    #for param in mdl.parameters():
    #    param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mdl.classifier.parameters(), lr=0.001)

    for epoch in range(n_epochs):  
        mdl.train().to(DEVICE)
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Epoch %s: " % (epoch), total=train_loader.__len__()):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = mdl(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        save(mdl, f'models/vgg16_finetuned_epoch{epoch}.pth')
        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.2}")
        wandb.log({'Train Loss': running_loss/len(train_loader)})

    return mdl


def evaluate_model(mdl, test_loader):
    mdl.eval() 
    preds = []
    y_true = []
    with torch.no_grad():  # No need to track gradients
        for inputs, labels in tqdm(test_loader):
            outputs = mdl(inputs.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    return y_true, preds


if __name__ == "__main__":

    # prepare args
    filename = "data/chestxpert/train_split_clean.csv"
    configs = {
        "n_labels": pd.read_csv(filename)["target"].nunique(),
        "n_epochs": 5,
        "batch_size": 128
    }

    # init wandb proj
    wandb.init(project="taecac",
               group="classifier",
               entity="gomes-inesisabel",
               job_type='vgg16',
               config=configs)

    # load datasets
    train = CheXpertDataset(csv_file= "data/chestxpert/train_split_clean.csv")
    train_loader = DataLoader(train, batch_size=configs["batch_size"], shuffle=True)

    test = CheXpertDataset(csv_file= "data/chestxpert/test_split_clean.csv")
    test_loader = DataLoader(test, batch_size=configs["batch_size"], shuffle=True)

    print("start training...")
    mdl = train_model(configs["n_epochs"], configs["n_labels"], train_loader)

    # save the model
    save(mdl, 'models/vgg16_finetuned.pth')

    print("start evaluation...")
    true_labels, predictions = evaluate_model(mdl, test_loader)

    wandb.log({"accuracy": accuracy_score(true_labels, predictions)})
    wandb.log({"classification_report": classification_report(true_labels, predictions)})
    wandb.log({"confusion_matrix": wandb.Image(sns.heatmap(confusion_matrix(true_labels, predictions)))})

    wandb.finish()
