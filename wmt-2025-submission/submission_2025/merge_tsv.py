
import pandas as pd
import sys
import pandas as pd

def merge_4_tsv_files(file1, file2, file3, file4, output_file):
    """
    Merges four TSV files with the same columns into one.
    
    Parameters:
        file1, file2, file3, file4 (str): Paths to the TSV files.
        output_file (str): Path to save the merged TSV file.
    """
    # Read all four TSV files
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    df3 = pd.read_csv(file3, sep='\t')
    df4 = pd.read_csv(file4, sep='\t')
    
    # Concatenate the DataFrames
    merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # Save the merged DataFrame to a new TSV file
    merged_df.to_csv(output_file, sep='\t', index=False)

    print(f"Merged file saved to: {output_file}")

    
if __name__ == "__main__":
    
    


    merge_4_tsv_files(file1, file2, file3, file4, output_file)
``