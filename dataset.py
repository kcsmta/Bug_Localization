import torch
from torch.utils.data import Dataset
import pandas as pd

class BugLocalizationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Args:
            csv_file (str): Path to the CSV file with bug localization data.
            tokenizer: The tokenizer instance, initialized in train.py.
            max_length (int): Maximum sequence length for tokenization.
        """
        # Load data from CSV file
        self.data = pd.read_csv(csv_file)
        
        # Set tokenizer and max length
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        # Return the number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a single data point
        sample = self.data.iloc[idx]
        
        # Prepare input text with <t_cls> token at the beginning
        input_text = "<t_cls> " + sample["buggy_code"]
        
        # Tokenize input text
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Extract input_ids and remove the extra batch dimension
        input_ids = inputs["input_ids"].squeeze()
        
        # Get start and end positions as labels
        start_pos = torch.tensor(sample["start_pos"], dtype=torch.long)
        end_pos = torch.tensor(sample["end_pos"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "start_pos": start_pos,
            "end_pos": end_pos
        }
