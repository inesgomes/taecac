import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torchvision.models import vgg16
from dataset import CheXpertDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from torch import load


DEVICE = "mps"

def plot_cluster_size(clusters, n_labels):
    """
    Plot the distribution of cluster sizes
    """
    plt.hist(clusters, bins=range(0, n_labels+1))
    plt.xlabel('Cluster')
    plt.ylabel('Number of data points')
    return plt


def plot_silhouette(X, labels):
    # Assuming 'X' is your data and 'labels' are your cluster labels
    silhouette_vals = silhouette_samples(X, labels)

    # Plot
    plt.figure(figsize=(10, 7))
    y_lower, y_upper = 0, 0
    #n_clusters = len(np.unique(labels))  # Number of clusters

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


def compute_tsne(X, labels):
    tsne = TSNE(n_components=2)
    reduced_data_tsne = tsne.fit_transform(X)
    plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=labels)
    return plt


if __name__ == "__main__":
    # some args
    dataset = "train_sml_clean"
    mdl_type = "pretrained" # or finetuned

    # configs
    filename = f"data/chestxpert/{dataset}.csv"
    configs = {
        "dataset": dataset,
        "n_image": len(pd.read_csv(filename)),
        "n_labels": pd.read_csv(filename)["target"].nunique(),
        "batch_size": 256
    }

    wandb.init(project="taecac",
               group="clustering",
               entity="gomes-inesisabel",
               job_type=f'vgg16{mdl_type}_pca_kmeans',
               config=configs)
    
    # load data for clustering
    dataset = CheXpertDataset(csv_file=filename)
    data_loader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=True)

    # load model
    if mdl_type == "pretrained":
        model = vgg16(pretrained=True).to(DEVICE)
    elif mdl_type == "finetuned":
        model = load('models/vgg16_finetuned.pth')

    # Extract features using the selected model
    features = []
    all_labels = []
    with torch.no_grad():  # No need to track gradients
        for images, labels in data_loader:
            output = model(images.to(DEVICE))
            output = output.view(output.size(0), -1).cpu().numpy()  # Flatten the features  # Move features to CPU and convert to numpy
            features.append(output.tolist()) 
            all_labels.extend(labels.tolist())#.cpu().numpy())

    wandb.log({"n_features": features.shape[1]})

    # Dimensionality Reductions
    pca = PCA(n_components=0.8, svd_solver='full', random_state=0)
    reduced_features = pca.fit_transform(features)

    wandb.log({"red_features": reduced_features.shape[1]})

    # Clustering
    kmeans = KMeans(n_clusters=configs["n_labels"], random_state=0)
    clusters = kmeans.fit_predict(reduced_features)

    # evaluate how good the clusters are
    # the higher the better
    wandb.log({"silhouette_score": silhouette_score(reduced_features, clusters)})
    # the higher the better
    wandb.log({"calinski_harabasz_score": calinski_harabasz_score(reduced_features, clusters)})
    # lower the better
    wandb.log({"davies_bouldin_score": davies_bouldin_score(reduced_features, clusters)})

    # evaluation
    # ONLY IF I HAVE THE TRUE LABELS 
    # simarity score - perfect if 1; 0 if random
    # wandb.log({"adjusted_rand_score": adjusted_rand_score(all_labels, clusters)})
    # wandb.log({"adjusted_mutual_info_score": adjusted_mutual_info_score(all_labels, clusters)})
    #vec_metrics = homogeneity_completeness_v_measure(all_labels, clusters)
    # wandb.log({"homogeneity": vec_metrics[0]})
    # wandb.log({"completeness": vec_metrics[1]})
    #wandb.log({"v_measure": vec_metrics[2]})
    #wandb.log({"fowlkes_mallows_score": fowlkes_mallows_score(all_labels, clusters)})

    # find distribution of clusters per label
    df = pd.DataFrame({"cluster": clusters, "label": all_labels})
    df_group = df.groupby("label", as_index=False).cluster.value_counts()
    df_agg = pd.merge(left=df_group,right=df.groupby("label").count().reset_index().rename(columns={"cluster": "sum"}), on="label") #.groupby("label").apply(lambda x: x/x.sum())
    df_agg["norm"] = df_agg["count"]/df_agg["sum"]
    df_count = df_agg.pivot(index="label", columns="cluster", values="norm").fillna(0)

    # some plots that are relevant
    wandb.log({"cluster_size": wandb.Image(plot_cluster_size(clusters, configs["n_labels"]))})
    wandb.log({"silhouette": wandb.Image(plot_silhouette(reduced_features, clusters))})
    # heatmap with percentage values and space between cells and bigger figure
    plt.figure(figsize=(10, 7))
    wandb.log({"confusion_matrix": wandb.Image(sns.heatmap(df_count, annot=True, cmap="Blues", fmt=".1%", linewidths=.5))})
    # visualization of the clusters
    plt.figure(figsize=(10, 7))
    wandb.log({"tsne": wandb.Image(compute_tsne(reduced_features, clusters))})

    wandb.finish()