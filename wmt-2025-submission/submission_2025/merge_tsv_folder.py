import os
import polars as pl

def merge_tsv_files(input_dir, output_filename="systems.tsv"):
    # Get all .tsv files in the input directory (no subdirectories)
    tsv_files = [f for f in os.listdir(input_dir) if f.endswith('.tsv') and os.path.isfile(os.path.join(input_dir, f))]
    
    if not tsv_files:
        print("No TSV files found in the directory.")
        return

    # List to collect DataFrames
    dataframes = []

    # Read and collect all TSV files
    for tsv_file in tsv_files:
        file_path = os.path.join(input_dir, tsv_file)
        try:
            df = pl.read_csv(file_path, separator='\t')
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Concatenate all dataframes
    merged_df = pl.concat(dataframes)

    # Output path
    output_path = os.path.join(input_dir, output_filename)

    # Save to a TSV file
    merged_df.write_csv(output_path, separator='\t')
    print(f"Merged {len(tsv_files)} files into {output_path}")

# Example usage
input_directory = "/home/sami/mmt-eval/doc-mte/submission_2025/roberta-secondary-2/system"
merge_tsv_files(input_directory)
