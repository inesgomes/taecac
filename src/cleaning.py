import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FOLDER = "data/chestxpert/"

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

def get_filename(split, only_diagnosis):
    diagnosis = "_onlydiagnosis" if only_diagnosis else ""
    return f"{FOLDER}{split}_split_clean{diagnosis}.csv"


if __name__ == "__main__":
    # some args
    sml = True
    only_diagnosis = True

    df_cleaned = (
      pd.read_csv(f"{FOLDER}train.csv")
      .pipe(cleaning)
      .pipe(create_target)
    )

    # rename path
    df_cleaned['path'] = df_cleaned["Path"].apply(lambda x: x.replace(f"CheXpert-v1.0-small/", FOLDER))

    # only relevant columns
    df_final = df_cleaned[["path", "target", "target_name"]]

    print(f"Number of images (total): {len(df_final):,}")

    if only_diagnosis:
        df_final = df_final[df_final["target_name"] != "No Finding"]
        print(f"Number of images with diagnosis: {len(df_final):,}")

    if sml:
        n_samples = 2000
        df_final = df_final.sample(2000, random_state=42)
        df_final.to_csv(f"{FOLDER}clean_n{n_samples}.csv", index=False)
    else:
        train, test = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final["target"])
        train.to_csv(get_filename("train", only_diagnosis), index=False)
        test.to_csv(get_filename("test", only_diagnosis), index=False)
