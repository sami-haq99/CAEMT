from datasets import load_dataset

import polars as pl
from random import randrange
from tqdm.auto import tqdm


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
  # Compute word counts 
  df_scored = df_paired.with_columns(( pl.col("src").str.len_chars() + pl.col("mt").str.len_chars()).alias("char_count_1"),
      (pl.col("src_pair").str.len_chars() + pl.col("mt_pair").str.len_chars()).alias("char_count_2")).with_columns(
      # Weighted score by character count
      (
          (pl.col("char_count_1") * pl.col("raw") + pl.col("char_count_2") * pl.col("raw_pair"))
          /
          (pl.col("char_count_1") + pl.col("char_count_2"))
      ).alias("raw_doc")
  )

  columns_to_keep = ['lp','src_doc', 'mt_doc', 'ref_doc', 'raw_doc', 'domain', 'year']
  df_scored = df_scored.select(columns_to_keep)
  df_scored = df_scored.rename({"src_doc": "src",
                                "mt_doc": "mt",
                                "ref_doc": "ref",
                                "raw_doc": "raw"})

  return df_scored


def process(df_scored, iterations):

  df_scored = df_scored.with_columns(sents = 1)
  df_scored = df_scored.drop("score", "annotators")

  dfs = [df_scored]
  count = 1

  for i in tqdm(range(iterations)):
    df_scored = pair_sentences(df_scored)
    count += 1
    df_scored = df_scored.with_columns(sents = count)
    dfs.append(df_scored)

  dfs_scored_concatenated = pl.concat(dfs)

  return dfs_scored_concatenated

dataset = load_dataset("../RicardoRei/WMT-mqm-human-evaluation")

df_train = dataset["train"].to_polars().drop_nulls()

#print(f"Unique language pairs:{df_train['lp'].unique().to_list()}")

print(f"Train: {df_train.shape[0]:,}")
#f_train_train = df_train.filter(pl.col("domain") == "news")
df_train_train = df_train.filter(pl.col("lp") == "en-ja")

print(f"Train: {df_train_train.shape[0]:,}")
#print(f"Domains: {df_train_train['domain'].unique().to_list()}")

df_scored_train = process(df_train_train, 5)


# show 3 random rows
#with pl.Config(set_fmt_str_lengths=300):
#  display(df_scored_test.sample(n=3))
  
from datasets import Dataset, DatasetDict

dataset_train = Dataset.from_polars(df_scored_train)
#dataset_test = Dataset.from_polars(df_scored_test)

dataset = DatasetDict({
    "train": dataset_train})

# Save the dataset
dataset.save_to_disk("./RicardoRei/wmt_mqm_long_text_augmentation_en_ja")
# Save the dataset as parquet  