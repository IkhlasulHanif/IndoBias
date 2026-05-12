import os
import pandas as pd

def export_training_sample(dataset, tokenizer, output_dir, step, num_samples=50):
    """
    Pulls a sample from the streaming dataset, decodes the token IDs, 
    and saves it to an Excel file to verify chunking and interleaving.
    """
    print(f"      -> 💾 Exporting {num_samples} training samples for sanity check...")
    try:
        samples = []
        # Create a fresh iterator from the streaming dataset
        dataset_iter = iter(dataset)
        
        for _ in range(num_samples):
            try:
                row = next(dataset_iter)
                # Decode input_ids back to text, keeping special tokens visible
                # so you can see exactly where sequences begin and end
                decoded_text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                
                samples.append({
                    "sequence_length": len(row['input_ids']),
                    "decoded_text": decoded_text
                })
            except StopIteration:
                break
                
        df = pd.DataFrame(samples)
        os.makedirs(output_dir, exist_ok=True)
        excel_filename = os.path.join(output_dir, f"training_sample_step_{step}.xlsx")
        df.to_excel(excel_filename, index=False)
        print(f"      ✅ Training sample successfully saved to: {excel_filename}")
        
    except Exception as e:
        print(f"      ❌ Failed to export training sample: {e}")