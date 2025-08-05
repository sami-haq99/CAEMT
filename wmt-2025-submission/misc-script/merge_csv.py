# Merge CSV files
import polars as pl
from random import randrange
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import os


#funtion to merge csv files with the same structure
def merge_csv_files(input_dir, output_file, subset="train"):
    #list all csv file which start with the subset name
    csv_files = [f for f in os.listdir(input_dir) if f.startswith(subset) and f.endswith('.csv')]
    # Initialize an empty list to hold DataFrames
    dataframes = []
    
    # Read each CSV file and append to the list
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        df = pl.read_csv(file_path)
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    merged_df = pl.concat(dataframes)
    
    # Save the merged DataFrame to a new CSV file
    merged_df.write_csv(output_file)
    print(f"Merged {len(csv_files)} files into {output_file}")

# Example usage
# read the directory containing CSV files
subset = "validation"  # Change this to the desired subset
input_directory = "../data-train2-da-sqm-mqm"
output_file = f"../data-train2-da-sqm-mqm/merged_2_augm_{subset}_sqm_mqm_da.csv"
merge_csv_files(input_directory, output_file, subset=subset)