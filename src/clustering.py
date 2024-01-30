import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import load
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision import transforms
import pandas as pd
import wandb
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from umap import UMAP

from tqdm import tqdm
from chexpert_dataset import CheXpertDataset
from aux import get_mean_std
from collections import Counter


DEVICE = "mps" if not torch.cuda.is_available() else "cuda:0"
FOLDER = "data/chestxpert/"

def plot_cluster_size(clusters):
    pcount = Counter(clusters)

    plt.barh(list(pcount.keys()), pcount.values())

    # Adding title and labels
    plt.xlabel('Number of elements per cluster')
    plt.ylabel('Cluster number')

    for element, frequency in pcount.items():
        plt.text(frequency, element ,f' {frequency:,}', va='center')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    return plt


def plot_silhouette(X, labels):
    # Assuming 'X' is your data and 'labels' are your cluster labels
    silhouette_vals = silhouette_samples(X, labels)

    # Plot
    plt.figure(figsize=(10, 7))
    y_lower, y_upper = 0, 0

    for _, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        plt.text(-0.03, (y_lower + y_upper) / 2, str(cluster))
        y_lower += len(cluster_silhouette_vals)

    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster label')
    plt.title('Silhouette Plot of Clustered Data')
    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")  # Average silhouette score 
    return plt

def plot_confusion_matrix(labels, clusters, axis):
    # find distribution of clusters per label
    df = pd.DataFrame({"cluster": clusters, "label": labels})
    df_group = df.groupby("label", as_index=False).cluster.value_counts()

    if axis == 0:
        df_agg = pd.merge(left=df_group,right=df.groupby("label").count().reset_index().rename(columns={"cluster": "sum"}), on="label")
    else:
        df_agg = pd.merge(left=df_group,right=df.groupby("cluster").count().reset_index().rename(columns={"label": "sum"}), on="cluster")


    df_agg["norm"] = df_agg["count"]/df_agg["sum"]
    df_count = df_agg.pivot(index="label", columns="cluster", values="norm").fillna(0)

    # create heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_count, annot=True, cmap="Blues", fmt=".1%", linewidths=.5)
    level = "row" if axis==0 else "column"
    plt.title(f"sum of percentages at the {level} level is 100%")
    return plt

def compute_tsne(X, labels):
    _, ax = plt.subplots(figsize=(10, 7))
    tsne = TSNE(n_components=2)
    reduced_data_tsne = tsne.fit_transform(X)
    ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=labels, s=1, alpha=0.7)
    ax.set_xlabel("t-SNE axis 1")
    ax.set_ylabel("t-SNE axis 2")
    ax.set_title("Plot of the clusters in two axis extracted from t-SNE")
    ax.legend()
    return plt


def get_features(data_loader, feature_extractor):
    features = []
    all_labels = []
    with torch.no_grad():  # No need to track gradients
        for images, _, labels in tqdm(data_loader):
            output = feature_extractor(images.to(DEVICE))
            features.append(output.detach().cpu().numpy())
            all_labels.append(labels)

    return np.concatenate(features, axis=0), np.concatenate(all_labels, axis=0)

if __name__ == "__main__":
    # some args
    dataset_name = "split_clean_onlydiagnosis"
    mdl_type = "pretrained" # pretrained or finetuned
    dim_red = "umap" # pca or umap
    clust = "kmeans" # kmeans or dbscan or gmm
    # do not forget the model name

    # configs
    filename = f"{FOLDER}test_{dataset_name}.csv"
    configs = {
        "dataset": dataset_name,
        "n_image": len(pd.read_csv(filename)),
        "n_labels": pd.read_csv(filename)["target"].nunique(),
        "batch_size": 256,
        "n_umap": 8,
    }

    wandb.init(project="taecac",
               group="clustering",
               entity="gomes-inesisabel",
               job_type=f'vgg16{mdl_type}_{dim_red}_{clust}',
               name="final",
               config=configs)
    
    # get data mean and std so that we can normalize before applying our algorithms
    # mean, std = get_mean_std(f"{FOLDER}train_{dataset_name}.csv")
    
    # prepare data
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # we need to rescale for the same size as vgg16 expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5062, 0.5062, 0.5062], std=[0.2873, 0.2873, 0.2873]),
    ])

    # load data
    #train_set = CheXpertDataset(csv_file=f"{FOLDER}train_{dataset_name}.csv", transform=preprocess)
    #train_loader = DataLoader(train_set, batch_size=configs["batch_size"], shuffle=True)
    test_set = CheXpertDataset(csv_file=f"{FOLDER}test_{dataset_name}.csv", transform=preprocess)
    test_loader = DataLoader(test_set, batch_size=configs["batch_size"], shuffle=True)

    # load model to extract embeddings
    if mdl_type == "pretrained":
        model = vgg16(pretrained=True).to(DEVICE)
    elif mdl_type == "finetuned":
        model = load(f'models/{dataset_name}/vgg16_finetuned_norm_v2_epoch0.pth').to(DEVICE)
    else:
        raise ValueError("mdl_type must be either pretrained or finetuned")

    # remove the classifier layer, so that we can access only the embedding
    model.classifier = torch.nn.Identity()
    
    print("start extracting features...")
    test_features, test_labels = get_features(test_loader, model)
    #train_features, train_labels = get_features(train_loader, model)
        
    wandb.log({"n_features": len(test_labels)})
    wandb.log({"silhouette_score_vgg16": silhouette_score(test_features, test_labels)})

    print("start dimensionality reduction...")
    if dim_red == "pca":
        red_alg = PCA(n_components=0.8, svd_solver='full', random_state=0)
        test_features = StandardScaler().fit_transform(test_features)
    elif dim_red == "umap":
        red_alg = UMAP(n_components=configs["n_umap"], random_state=0)
    else:
        raise ValueError("dim_red must be either pca or umap")

    test_reduced_features = red_alg.fit_transform(test_features)

    wandb.log({"red_features": test_reduced_features.shape[1]})
    wandb.log({"silhouette_score_reduction": silhouette_score(test_reduced_features, test_labels)})
    
    print("start clustering...")
    if clust == "dbscan":
        cl_alg = DBSCAN(eps=0.3, min_samples=10)
    elif clust == "kmeans":
        cl_alg = KMeans(n_clusters=configs["n_labels"], random_state=0)
    elif clust == "gmm":
        cl_alg = GaussianMixture(n_components=configs["n_labels"], random_state=0)
    else:
        raise ValueError("clustering must be either dbscan or kmeans")
    
    clusters = cl_alg.fit_predict(test_reduced_features)

    print("evaluation")
    # evaluate how good the clusters are
    # the higher the better
    wandb.log({"silhouette_score": silhouette_score(test_reduced_features, clusters)})
    # the higher the better
    wandb.log({"calinski_harabasz_score": calinski_harabasz_score(test_reduced_features, clusters)})
    # lower the better
    wandb.log({"davies_bouldin_score": davies_bouldin_score(test_reduced_features, clusters)})

    # some plots that are relevant
    wandb.log({"cluster_size": wandb.Image(plot_cluster_size(clusters))})
    wandb.log({"silhouette": wandb.Image(plot_silhouette(test_reduced_features, clusters))})
    # heatmap with percentage values and space between cells and bigger figure
    wandb.log({"confusion_matrix_col": wandb.Image(plot_confusion_matrix(test_labels, clusters, axis=0))})
    wandb.log({"confusion_matrix_row": wandb.Image(plot_confusion_matrix(test_labels, clusters, axis=1))})  
    # contigency matrix
    plt.figure(figsize=(10, 7))
    wandb.log({"contingency_matrix": wandb.Image(sns.heatmap(contingency_matrix(test_labels, clusters), annot=True, fmt=".1%", linewidths=1))})

    # visualization of the clusters
    wandb.log({"tsne": wandb.Image(compute_tsne(test_reduced_features, clusters))})

    wandb.finish()
