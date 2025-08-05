from datasets import load_dataset

import polars as pl
from random import randrange
from tqdm.auto import tqdm




dataset = load_dataset("../../RicardoRei/WMT-en-ru-augment-train-DA")

df_train = dataset["train"].to_polars().drop_nulls()

#print(f"Unique language pairs:{df_train['lp'].unique().to_list()}")

print(f"Train: {df_train.shape[0]:,}")
#f_train_train = df_train.filter(pl.col("domain") == "news")

print(f"Train: {df_train.shape[0]:,}")
print(f"Domains: {df_train['domain'].unique().to_list()}")
print(f"Lp: {df_train['lp'].unique().to_list()}")
print(f"Score: {df_train['raw'].unique().to_list()}")

#df_filtered = df_train.filter(pl.col("sents") != 1)
#print("Rows where 'sents' column has value other than 1:")
print(df_train.head(5))

import matplotlib.pyplot as plt

# Filter rows where 'raw' column values are between 0 and 100
#df_filtered_raw = df_train.filter((pl.col("raw") >= 0) & (pl.col("raw") <= 100))

# Extract 'raw' column values
raw_values = df_train["raw"].to_list()

# Plot the distribution of 'raw' values
plt.figure(figsize=(10, 6))
plt.hist(raw_values, bins=20, color='blue', alpha=0.7)
plt.title("Distribution of 'raw' Column Values")
plt.xlabel("Raw Values")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
#plt.show()
plt.savefig("raw_DA_distribution.png")
#df_train_norm = df_train

# Normalize the 'raw' column

