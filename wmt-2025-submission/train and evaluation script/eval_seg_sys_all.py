import os
import torch
import csv
from COMET.comet.models.regression.regression_metric import RegressionMetric
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

#create a main function to run the code
if __name__ == "__main__":

    # take the language pair, model name and model_path as a command line argument
    if len(sys.argv) < 4:
        print("Usage: python eval_csv_to_tsv_all.py <language_pair> <model_name> <model_path>")
        sys.exit(1)
    lang = sys.argv[1]
    model_name = sys.argv[2]
    model_path = sys.argv[3]

    #model_path = "../lightning_logs/mqm/checkpoints/epoch=4-step=107255-val_kendall=0.843.ckpt"
    model = RegressionMetric.load_from_checkpoint(model_path)
    model.eval()

    print(f"Language pair, model name, model path: {lang}, {model_name}, {model_path}")

    # === Load the model checkpoint ===
    # Load the data from the CSV file
    dir_name = '/home/shaq/mtqe/wmt-esa-2025'
    method = 'esa'
    output_dir = '/home/shaq/mtqe/submission_2025'
    #model_name = 'mqm'
    gpus = 2  # Number of GPUs to use
    #run the below code for each language pair

    file_name = f"{lang}.csv"
    file_path = os.path.join(dir_name, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Please check the path.")
        exit(1)

    
    print(f"Processing file: {file_path}")
    data = read_csv(file_path)
    


    #get the 'source_segment' column from the data and save it to a list
    source_segments = [row['source_segment'] for row in data if 'source_segment' in row]    
    reference_segments = [row['reference_segment'] for row in data if 'reference_segment' in row]
    mt_segments = [row['hypothesis_segment'] for row in data if 'hypothesis_segment' in row]

    # createa list of dictionaries with each row containing 'src', 'ref', and 'mt' segments

    segments = [{'src': src, 'ref': ref, 'mt': mt} for src, ref, mt in zip(source_segments, reference_segments, mt_segments) if src and ref and mt] 

    with torch.no_grad():
        model_output = model.predict(segments, batch_size=16, gpus=gpus)


    #create a TSV file with the colums from data 'doc__id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id', 'domain_name', 'method', 'overall'
    output_tsv_file = f'{output_dir}/{model_name}_{method}_{lang}_scores.tsv'
    with open(output_tsv_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        # Write the header
        writer.writerow(['doc_id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id', 'domain_name', 'method', 'overall'])
        
        # Write the data
        for i, row in enumerate(data):
            if i < len(model_output["scores"]):  # Ensure we have a score for this row
                score = model_output["scores"][i]
                score = min(1, max(0, score))
                writer.writerow([
                    row.get('doc_id', ''),
                    row.get('segment_id', ''),
                    row.get('source_lang', ''),
                    row.get('target_lang', ''),
                    row.get('set_id', ''),
                    row.get('system_id', ''),
                    row.get('domain_name', ''),
                    row.get('method', ''),
                    #overall is the score from the model output,
                    f"{score:.2f}"  # Format the score to 4 decimal places
                ])
    print(f"Results for {lang} saved to {output_tsv_file}")
    
    #saveing system_scores.tsv
    system_scores_file = f'{output_dir}/{model_name}_{method}_{lang}_system_scores.tsv'
    with open(system_scores_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        # Write the header
        system_score = model_output['system_score']
        writer.writerow(['doc_id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id', 'domain_name', 'method', 'overall'])
                
        writer.writerow([
                    '', 'all',
                    '', 'all',
                    row.get('source_lang', ''),
                    row.get('target_lang', ''),
                    row.get('set_id', ''),
                    row.get('system_id', ''),
                    '', 'all',
                    row.get('method', ''),
                    #overall is the score from the model output,
                    f"{system_score:.2f}"  # Format the score to 4 decimal places
                ])
    print(f"System scores for {lang} saved to {system_scores_file}")
