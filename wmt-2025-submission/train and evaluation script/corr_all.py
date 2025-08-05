import os
import torch
import csv
from COMET.comet.models.regression.regression_metric import RegressionMetric
from scipy.stats import pearsonr
import sys

# === Load your trained checkpoint ===

#read a tsv file and return a list of dictionaries
def read_csv(file_path):
    #read data from a CSV file
    data = []
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return data
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return data


if __name__ == "__main__":

    # take the language pair and model path as a command line argument
    
    if len(sys.argv) < 6:
        print("Usage: python eval_csv_to_tsv_all.py <language_pair> <model_name> <model_path> <dir_name>")
        sys.exit(1)
    lang = sys.argv[1]
    model_name = sys.argv[2]
    model_path = sys.argv[3]
    dir_name = sys.argv[4]
    method = sys.argv[5]
    print(f"Language pair: {lang}, Model name: {model_name}, Model path: {model_path}, Directory name: {dir_name}")

    # Load the trained model
    # Make sure to adjust the path to your model checkpoint
    # Example: model_path = "path/to/your/model.ckpt"
    #model_path ="../lightning_logs/mqm/checkpoints/epoch=4-step=107255-val_kendall=0.843.ckpt"
    model = RegressionMetric.load_from_checkpoint(model_path)
    model.eval()


    # Load the data from the CSV file
    #method = 'sqm'
    gpus = 2
    #run the below code 
    #for each language pair


    #file_name = f"test-{lang}-{method}.csv"
    file_name = f"test_all.csv"
    file_path = os.path.join(dir_name, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Please check the path.")

    data = read_csv(file_path)

    #get the 'source_segment' column from the data and save it to a list
    source_segments = [row['src'] for row in data if 'src' in row]    
    reference_segments = [row['ref'] for row in data if 'ref' in row]
    mt_segments = [row['mt'] for row in data if 'mt' in row]
    ref_score = [row['score'] for row in data if 'score' in row]

    # createa list of dictionaries with each row containing 'src', 'ref', and 'mt' segments
    segments = [{'src': src, 'ref': ref, 'mt': mt} for src, ref, mt in zip(source_segments, reference_segments, mt_segments) if src and ref and mt] 

    with torch.no_grad():
        model_output = model.predict(segments, batch_size=16, gpus=gpus)


    ref_score = [float(x) for x in ref_score]
    model_score = [float(x) for x in model_output.get("scores", [])[:len(ref_score)]]
    for i in range(min(5, len(ref_score))):
        print(f"Ref: {ref_score[i]:.2f}, Model: {model_score[i]:.2f}")

    # Optional: check length match
    if len(ref_score) != len(model_score):
        raise ValueError("ref_score and model_score must be of the same length")

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(ref_score, model_score)

    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    # Save the results to a CSV file
    output_file = f"pcorr-{lang}-{method}-{model_name}.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['p-value', 'correlation'])
        writer.writerow([p_value, correlation])
    print(f"Results saved to {output_file}")
