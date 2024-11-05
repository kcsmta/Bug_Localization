import torch
from torch import nn
from transformers import T5ForConditionalGeneration, RobertaTokenizer

class BugLocalizationModel(nn.Module):
    def __init__(self, model, tokenizer, hidden_size):
        super(BugLocalizationModel, self).__init__()
        
        # Load T5 model and get encoder
        self.model = model
        self.tokenizer = tokenizer
                
        # Define projection layers for start and end position attention queries
        self.W_q_start = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_q_end = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids):
        # Add the <t_cls> token at the start of input_ids
        t_cls_token_id = self.tokenizer.convert_tokens_to_ids("<t_cls>")
        t_cls_input = torch.full((input_ids.size(0), 1), t_cls_token_id, device=input_ids.device)
        input_ids = torch.cat([t_cls_input, input_ids], dim=1)
        
        # Pass inputs through the encoder
        encoder_outputs = self.model(input_ids=input_ids)
        hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Step 1: Use <t_cls> embedding to find start position
        t_cls_embedding = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        query_start = self.W_q_start(t_cls_embedding)  # Project <t_cls> embedding to start query

        # Project hidden states to keys for attention
        keys = self.W_k(hidden_states)  # Shape: (batch_size, seq_len, hidden_size)
        attention_scores_start = torch.matmul(query_start.unsqueeze(1), keys.transpose(-1, -2)).squeeze(1)  # (batch_size, seq_len)
        attention_probs_start = torch.softmax(attention_scores_start, dim=-1)
        
        # Predicted start position
        start_pos = torch.argmax(attention_probs_start, dim=1)  # Shape: (batch_size,)

        # Step 2: Use start position embedding to find end position
        start_embeddings = hidden_states[torch.arange(hidden_states.size(0)), start_pos, :]  # Shape: (batch_size, hidden_size)
        query_end = self.W_q_end(start_embeddings)  # Project start embedding to end query

        attention_scores_end = torch.matmul(query_end.unsqueeze(1), keys.transpose(-1, -2)).squeeze(1)  # (batch_size, seq_len)
        attention_probs_end = torch.softmax(attention_scores_end, dim=-1)
        
        # Predicted end position
        end_pos = torch.argmax(attention_probs_end, dim=1)  # Shape: (batch_size,)

        return attention_probs_start, attention_probs_end
