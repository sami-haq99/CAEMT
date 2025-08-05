from datasets import load_dataset

import polars as pl
from random import randrange
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt



dataset = load_dataset("/home/sami/mmt-eval/doc-mte/RicardoRei/WMT-en-ru-augment-train-sqm")

df_train = dataset["train"].to_polars().drop_nulls()

#print(f"Unique language pairs:{df_train['lp'].unique().to_list()}")

print(f"Train: {df_train.shape[0]:,}")
#f_train_train = df_train.filter(pl.col("domain") == "news")

print(f"Train: {df_train.shape[0]:,}")
print(f"Domains: {df_train['domain'].unique().to_list()}")


print(df_train.head(5))

mean_score = df_train["raw"].mean()
median_score = df_train["raw"].median()
mode_score = mode(df_train["raw"])

# Plot
plt.figure(figsize=(10, 6))
plt.hist(df_train["raw"], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
plt.axvline(median_score, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_score:.2f}')
# plt.axvline(mode_score, color='blue', linestyle='dashed', linewidth=1, label=f'Mode: {mode_score:.2f}')



# Optional: Use log scale for better visibility of rare bins
# plt.yscale('log')

plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.yscale('log')

plt.show()