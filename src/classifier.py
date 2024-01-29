import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch import save, load
import wandb
import pandas as pd
from chexpert_dataset import CheXpertDataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# TODO solve problem of unbalanced classes

DEVICE = "mps" if not torch.cuda.is_available() else "cuda:1"
FOLDER = "data/chestxpert/"

def train_model(n_epochs, n_labels, dataset_name, run_name, train_loader):
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
        for inputs, labels, _ in tqdm(train_loader, desc="Epoch %s: " % (epoch), total=train_loader.__len__()):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = mdl(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        save(mdl, f'models/{dataset_name}/vgg16_finetuned_{run_name}_epoch{epoch}.pth')
        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.2}")
        wandb.log({'Train Loss': running_loss/len(train_loader)})
    return mdl


def evaluate_model(mdl, test_loader):
    mdl.eval() 
    preds = []
    y_true = []
    with torch.no_grad():  # No need to track gradients
        for inputs, labels, _ in tqdm(test_loader):
            outputs = mdl(inputs.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted.tolist())
            y_true.append(labels.tolist())
    return np.concatenate(y_true, axis=0), np.concatenate(preds, axis=0)


def get_n_labels(filename):
    return pd.read_csv(f"{FOLDER}train_{filename}.csv")["target"].nunique()

if __name__ == "__main__":

    # prepare args
    dataset_name = "split_clean_onlydiagnosis"
    run_name = "v1"
    configs = {
        "n_labels": get_n_labels(dataset_name),
        "n_epochs": 3,
        "batch_size": 64,
        "dataset": dataset_name
    }

    # init wandb proj
    wandb.init(project="taecac",
               group="classifier",
               entity="gomes-inesisabel",
               job_type='vgg16',
               name=f"{dataset_name}_{run_name}",
               config=configs)

    # load datasets
    train = CheXpertDataset(csv_file=f"{FOLDER}/train_{dataset_name}.csv")
    train_loader = DataLoader(train, batch_size=configs["batch_size"], shuffle=True)

    test = CheXpertDataset(csv_file= f"{FOLDER}/test_{dataset_name}.csv")
    test_loader = DataLoader(test, batch_size=configs["batch_size"], shuffle=True)

    #print("start training...")
    #mdl = train_model(configs["n_epochs"], configs["n_labels"], dataset_name, run_name, train_loader)

    # save the model
    #save(mdl, f'models/{dataset_name}/vgg16_finetuned_{run_name}.pth')
    mdl = load(f'models/{dataset_name}/vgg16_finetuned_{run_name}.pth')

    print("start evaluation...")
    true_labels, predictions = evaluate_model(mdl, test_loader)

    wandb.log({"accuracy": accuracy_score(true_labels, predictions)})
    wandb.log({"f1_micro": f1_score(true_labels, predictions, average="micro")})
    wandb.log({"f1_macro": f1_score(true_labels, predictions, average="macro")})
    wandb.log({"f1_weighted": f1_score(true_labels, predictions, average="weighted")})
    wandb.log({"precision_micro": precision_score(true_labels, predictions, average="micro")})
    wandb.log({"precision_macro": precision_score(true_labels, predictions, average="macro")})
    wandb.log({"precision_weighted": precision_score(true_labels, predictions, average="weighted")})
    wandb.log({"recall_micro": recall_score(true_labels, predictions, average="micro")})
    wandb.log({"recall_macro": recall_score(true_labels, predictions, average="macro")})
    wandb.log({"recall_weighted": recall_score(true_labels, predictions, average="weighted")})
    # wandb.log({"roc_auc": roc_auc_score(true_labels, predictions, multi_class='ovr', average="weighted")})
    print(classification_report(true_labels, predictions))

    wandb.finish()
