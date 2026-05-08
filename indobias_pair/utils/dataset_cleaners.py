import re

def clean_liputan6_text(examples):
    """
    Cleans structural and scraping artifacts from the Liputan6 dataset.
    Takes a batch of examples with a 'document' column and outputs a 'text' column.
    """
    cleaned_texts = []
    
    for text in examples["document"]:
        # Strip the location header
        text = re.sub(r'^Liputan6 \. com , [A-Za-z\s]+ : ', '', text)
        
        # Remove the inline "[ baca : ... ]" links
        text = re.sub(r'\s*\[\s*baca\s*:.*?\]', '', text, flags=re.IGNORECASE)
        
        # Remove the reporter sign-offs at the very end
        text = re.sub(r'\s*\([^)]+\)\s*\.?\s*$', '.', text)
        
        # Repair shattered numbers and times (e.g., "1 . 500" -> "1.500")
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        
        # Fix detached punctuation
        text = re.sub(r'\s+([.,?!:;])', r'\1', text)
        
        # Fix floating quotation marks
        text = re.sub(r'(["“])\s+', r'\1', text)
        text = re.sub(r'\s+(["”])', r'\1', text)
        
        # Compress multiple spaces into one and trim edges
        text = re.sub(r'\s{2,}', ' ', text).strip()
        
        cleaned_texts.append(text)
        
    # Return standard 'text' column for the pipeline
    return {"text": cleaned_texts}