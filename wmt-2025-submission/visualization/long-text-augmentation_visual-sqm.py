from datasets import load_dataset

import polars as pl
from random import randrange
from tqdm.auto import tqdm




dataset = load_dataset("../../RicardoRei/WMT-en-ja-augment-train-sqm")

df_train = dataset["train"].to_polars().drop_nulls()

#print(f"Unique language pairs:{df_train['lp'].unique().to_list()}")

print(f"Train: {df_train.shape[0]:,}")
#f_train_train = df_train.filter(pl.col("domain") == "news")

print(f"Train: {df_train.shape[0]:,}")
print(f"Domains: {df_train['domain'].unique().to_list()}")
print(f"Lp: {df_train['lp'].unique().to_list()}")
print(f"Score: {df_train['raw'].unique().to_list()}")

#print some examples rows where value of 'sents' is greater than 1
print("Examples of rows with 'sents' > 1:")
df_train.filter(pl.col("sents") > 1).head(5).write_csv("examples_3_sents.csv")
print(df_train.filter(pl.col("sents") == 3).head(5))
# Visualize the distribution of 'raw' column values 

import matplotlib.pyplot as plt

# Extract 'raw' column values
raw_values = df_train["raw"].to_list()

# Plot the distribution of 'raw' values
plt.figure(figsize=(10, 6))
plt.hist(raw_values, bins=20, color='blue', alpha=0.7)
plt.title("Distribution of 'raw' Column Values")
plt.xlabel("Raw Values")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

