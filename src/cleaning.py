import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def cleaning(df):
    # drop all rows with -1 in any column -> uncertain
    mask_uncertain = (df[df == -1].sum(axis=1)==0)
    # only frontal x-rays
    mask_frontal = (df["Frontal/Lateral"] == "Frontal")

    df_final = df[mask_uncertain & mask_frontal]

    # only one problem detected
    mask_one_label = df_final[df_final.columns[5:-1]].sum(axis=1)==1
    return df_final[mask_one_label]


def create_target(df):
    # rename for multiclass classification
    df["target_name"] = df[df.columns[5:-1]].apply(lambda x: x[x==1].index.tolist()[0], axis=1)
    # target number
    df["target"] = LabelEncoder().fit_transform(df["target_name"])
    return df


if __name__ == "__main__":
    # some args
    folder = "data/chestxpert/"
    filename = "train"
    sml = False

    df_cleaned = (
      pd.read_csv(f"{folder}{filename}.csv")
      .pipe(cleaning)
      .pipe(create_target)
    )

    # rename path
    df_cleaned['path'] = df_cleaned["Path"].apply(lambda x: x.replace(f"CheXpert-v1.0-small/", folder))

    # only relevant columns
    df_final = df_cleaned[["path", "target", "target_name"]]

    print(f"Number of images: {len(df_final):,}")

    if sml:
        df_final = df_final.sample(5000, random_state=42)
        filename += "_sml"
        df_final.to_csv(f"{folder}{filename}_clean.csv", index=False)
    else:
        train, test = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final["target"])
        train.to_csv(f"{folder}train_split_clean.csv", index=False)
        test.to_csv(f"{folder}test_split_clean.csv", index=False)
