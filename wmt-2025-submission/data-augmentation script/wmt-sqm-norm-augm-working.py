import polars as pl
from random import randrange
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict

from datasets import load_dataset
import os


def pair_sentences(df):
    df1 = df.with_row_index()
    df2 = df1.sample(fraction=1, shuffle=True)

    group_columns = ["lp", "domain"]

    # Group by language and domain
    df1_groups = df1.group_by(group_columns).agg(pl.all()).sort(group_columns)
    df2_groups = df2.group_by(group_columns).agg(pl.all()).sort(group_columns)

    df_joined = df1_groups.join(df2_groups,
                                on=group_columns,
                                how="left",
                                suffix="_pair")

    df_paired = df_joined.explode(pl.exclude(group_columns))

    # Filter out exact indices
    df_paired = df_paired.filter(pl.col("index") != pl.col("index_pair"))

    # Combine the sentences in each pair
    df_paired = df_paired.with_columns([
        (pl.col("src") + " " + pl.col("src_pair")).alias("src_doc"),
        (pl.col("ref") + " " + pl.col("ref_pair")).alias("ref_doc"),
        (pl.col("mt") + " " + pl.col("mt_pair")).alias("mt_doc"),
    ])

    # Compute weighted average score
    df_scored = df_paired.with_columns(
        (
            pl.col("src").str.len_chars() + pl.col("mt").str.len_chars()
        ).alias("char_count_1"),
        (
            pl.col("src_pair").str.len_chars() + pl.col("mt_pair").str.len_chars()
        ).alias("char_count_2")
    ).with_columns(
        (
            (pl.col("char_count_1") * pl.col("raw") + pl.col("char_count_2") * pl.col("raw_pair"))
            /
            (pl.col("char_count_1") + pl.col("char_count_2"))
        ).alias("raw_doc")
    )

    columns_to_keep = ['lp', 'src_doc', 'mt_doc', 'ref_doc', 'raw_doc', 'domain', 'year']
    df_scored = df_scored.select(columns_to_keep)
    df_scored = df_scored.rename({"src_doc": "src",
                                  "mt_doc": "mt",
                                  "ref_doc": "ref",
                                  "raw_doc": "raw"})

    return df_scored


def process(df_scored, iterations):
    
    score_min = df_scored.select(pl.col("raw").min()).item()
    score_max = df_scored.select(pl.col("raw").max()).item()

    df_scored = df_scored.with_columns(
        ((pl.col("raw") - score_min) / (score_max - score_min)).alias("raw")
    )
    
    df_scored = df_scored.with_columns(sents=1)
    df_scored = df_scored.drop("system", "annotators", strict=False)

    dfs = [df_scored]
    count = 1

    for i in tqdm(range(iterations)):
        df_scored = pair_sentences(df_scored)
        count += 2
        df_scored = df_scored.with_columns(sents=count)
        dfs.append(df_scored)

    dfs_scored_concatenated = pl.concat(dfs)

    return dfs_scored_concatenated


lang_pair = ["en-cs", "en-ru", "en-zh", "en-ja"]  # Change this to the desired language pair
for lp in lang_pair:
    print(f"Processing language pair: {lp}")
    subset = "sqm"
    dataset = load_dataset("../RicardoRei/WMT-" + subset + "-human-evaluation")

    df_train = dataset["train"].to_polars().drop_nulls()
    # Filter for the specific language pair
    if lp not in df_train["lp"].unique().to_list():
        raise ValueError(f"Language pair {lp} not found in the dataset.")
    df_train = df_train.filter(pl.col("lp") == lp)

    #since the DA data has raw scores, we rename the column to match the expected format
    df_train = df_train.rename({"score": "raw"})
    df_scored_train = process(df_train, 2)

    print(df_scored_train.null_count())

    # Save data as HuggingFace Dataset
    dataset_train = Dataset.from_polars(df_scored_train)
    dataset = DatasetDict({
        "train": dataset_train})

    # Save the dataset
    output_dir = "../RicardoRei/WMT-" + lp + "-augment-train-" + subset
    os.makedirs(output_dir, exist_ok=True)
    df_scored_train.write_csv(f"{output_dir}/train.csv")
