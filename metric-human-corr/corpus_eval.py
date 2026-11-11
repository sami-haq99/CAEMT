import torch
import numpy as np
import pandas as pd
from typing import List, Union
from tqdm import tqdm
from similarity_score_mme import JinaV4SimilarityMapper # Your local file
import os
class CorpusScorer(JinaV4SimilarityMapper):
    """
    Optimized scorer for generating individual row-level results on a GPU.
    """
    
    def calculate_individual_scores(
        self, 
        sources: List[str], 
        candidates: List[str], 
        images: List[str], # List of file paths
        batch_size: int = 8,
        output_file: str = "corpus_results.csv"
    ):
        results_data = []
        total = len(sources)
        
        print(f"🚀 Starting individual scoring for {total} samples on {self.device}...")
        
        # We use no_grad because we are not training, this saves massive VRAM
        with torch.no_grad():
            
            for i in tqdm(range(0, total, batch_size), desc="Processing Batches"):
                # 1. Slicing the Batch
                b_src = sources[i : i + batch_size]
                b_tgt = candidates[i : i + batch_size]
                b_img_paths = images[i : i + batch_size]
                
                # 2. Bulk Encoding (The heavy lifting)
                # Text: [Batch_Size] -> List of Tensors
                vecs_src = self.model.encode_text(
                    b_src, task=self.task, prompt_name="query", return_multivector=True
                )
                vecs_tgt = self.model.encode_text(
                    b_tgt, task=self.task, prompt_name="query", return_multivector=True
                )
                
                # Image: Load -> Resize -> Encode
                # We load images here to ensure they match the batch index
                b_pil_imgs = []
                valid_indices = [] # Track valid images in case of load errors
                
                for idx, img_path in enumerate(b_img_paths):
                    try:
                        # Use your class's internal loader to get consistent resizing
                        img = self._load_image(img_path) 
                        b_pil_imgs.append(img)
                        valid_indices.append(idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        # Push a placeholder (black image) to keep indices aligned
                        b_pil_imgs.append(Image.new('RGB', (1024, 1024)))

                vecs_img = self.model.encode_image(
                    b_pil_imgs, 
                    task=self.task, 
                    return_multivector=True,
                    max_pixels=1024*1024,
                    truncate_dim=self.num_vectors
                )

                # 3. Pairwise Scoring
                # We iterate through the batch index 'k' to score (Src[k] vs Tgt[k])
                # This is fast because the vectors are already on GPU
                for k in range(len(b_src)):
                    # Get Tensors
                    v_s = torch.tensor(vecs_src[k]).to(self.device)
                    v_t = torch.tensor(vecs_tgt[k]).to(self.device)
                    v_i = torch.tensor(vecs_img[k]).to(self.device)
                    
                    # A. Compute Raw MaxSims
                    s_fidelity = self._calc_single_maxsim(v_s, v_t) # Text-Text
                    s_grounding = self._calc_single_maxsim(v_t, v_i) # Text-Image
                    s_relevance = self._calc_single_maxsim(v_s, v_i) # Source-Image
                    
                    # B. Apply Formula
                    # Gate: lambda = max(0, relevance)^2
                    k_factor = 2
                    lamb = max(0, s_relevance) ** k_factor

                    # Weighted Formula
                    final_score = (s_fidelity + (lamb * s_grounding)) / (1 + lamb + 1e-9)

                    # 4. Store Row
                    results_data.append({
                        "id": i + k,
                        "source": b_src[k],
                        "candidate": b_tgt[k],
                        "image_path": b_img_paths[k],
                        "Final_Weighted_Score": round(final_score, 4),
                        "Text_Fidelity": round(s_fidelity, 4),
                        "Visual_Grounding": round(s_grounding, 4),
                        "Image_Relevance": round(s_relevance, 4)
                    })

        # 5. Save to CSV
        df = pd.DataFrame(results_data)
        df.to_csv(output_file, index=False)
        print(f"✅ Finished! Results saved to {output_file}")
        return df

    def _calc_single_maxsim(self, t1, t2):
        """Helper to calculate MaxSim between two tensors on GPU"""
        t1 = torch.nn.functional.normalize(t1, p=2, dim=1)
        t2 = torch.nn.functional.normalize(t2, p=2, dim=1)
        sim_matrix = torch.matmul(t1, t2.T)
        return sim_matrix.max(dim=1).values.mean().item()



def load_corpus(src_file, tgt_file, image_file, image_dir):

    sources = []
    candidates = []
    images = []

    with open(src_file, 'r') as f:
        sources.extend([line.strip() for line in f])
    with open(tgt_file, 'r') as f:
        candidates.extend([line.strip() for line in f])
    with open(image_file, 'r') as f:
        #add full path to file name
        images.extend([os.path.join(image_dir, line.strip()) for line in f])

    # Load your corpus data here
    return sources, candidates, images



# --- RUNNER ---
if __name__ == "__main__":
    from PIL import Image # Ensure PIL is available
    
    # 1. Setup Data (Example)
    # In real usage, load these from a file
    
    mapper = JinaV4SimilarityMapper(task = 'retrieval') 
    DATA_DIR = "../data/comute/"
    IMAGE_DIR = "../data/comute/images/"  # Directory containing images
    l_pair = ["en-cs", "en-ar", "en-zh", "en-fr","en-de"]

         # e.g. "ar" to keep one language only; None = keep everything

    for lang in l_pair:
        FILE = f"aggregated_scores_{lang}.csv"
        df = pd.read_csv(DATA_DIR + FILE, encoding="utf-8-sig")   # utf-8-sig handles the Arabic text and any BOM
        df.columns = df.columns.str.strip()


        # the same sys + seg should not appear twice, otherwise one entry would overwrite the other
        dups = df[df.duplicated(["sys", "seg"], keep=False)]
        if not dups.empty:
            print(f"WARNING: {len(dups)} rows share a (sys, seg) pair (multiple languages?). "
                "Set LANG to keep one language.")

        # build the nested dictionary
        aggr_data = {}
        for r in df.to_dict(orient="records"):
            entry = {k: v for k, v in r.items() if k not in ("sys", "seg")}
            aggr_data.setdefault(r["sys"], {})[int(r["seg"])] = entry

        # read the srctxt, tgttxt, and img from the aggregated data for each system and segment, and save them to a new CSV file for scoring

        output_dir = "scoring_results/"
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for sys, seg_dict in aggr_data.items():
            
            for seg, entry in seg_dict.items():
                src = entry["srctxt"]
                cand = entry["tgttxt"]
                img = entry["img"]
                # calculate the scores for each record
                result = mapper.calculate_multimodal_consistency(src, cand, os.path.join(IMAGE_DIR, img))
                result.update({'seg': seg, 'system': sys, 'source': src, 'candidate': cand, 'image_path': img})
                results.append(result)

        with open(f"{output_dir}{lang}_mmss_results.csv", "w") as f:
                #add the header column
                print(f"Writing results to {output_dir}_{lang}_results.csv")
                f.write("Segment, System, Source, Candidate, Image, Final_Score, Text_Fidelity (Src-Tgt), Visual_Grounding (Tgt-Img), Image_Relevance (Src-Img)\n")
                for i in range(len(results)):
                    #remove the image path from images
                    #The csv source and candidates also have commas, so we need to wrap them in quotes
                    f.write(f"\"{results[i]['seg']}\",\"{results[i]['system']}\",\"{results[i]['source']}\",\"{results[i]['candidate']}\",\"{results[i]['image_path']}\",\"{results[i]['Final_Score']}\",\"{results[i]['Text_Fidelity (Src-Tgt)']}\",\"{results[i]['Visual_Grounding (Tgt-Img)']}\",\"{results[i]['Image_Relevance (Src-Img)']}\"\n")

