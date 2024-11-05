import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from models import BugLocalizationModel  # Assuming you have this custom model defined in models.py
from dataset import BugLocalizationDataset  # Assuming the custom dataset is saved in dataset.py

# File paths and model setup
training_file = 'train_tufano_small.csv'
model_checkpoint = 'Salesforce/codet5-large'

# Load model and tokenizer
base_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).get_encoder()  # Load only the encoder part
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({"additional_special_tokens": ["<t_cls>"]})

base_model.resize_token_embeddings(len(tokenizer))

hidden_size = base_model.config.d_model

model = BugLocalizationModel(base_model, tokenizer, hidden_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Initialize the dataset and DataLoader
train_dataset = BugLocalizationDataset(csv_file=training_file, tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch in train_dataloader:
        # Move batch data to device
        input_ids = batch["input_ids"].to(device)
        start_pos = batch["start_pos"].to(device)
        end_pos = batch["end_pos"].to(device)
        
        # Forward pass through the model
        start_logits, end_logits = model(input_ids=input_ids)
        
        # Calculate losses for start and end positions
        loss_start = criterion(start_logits, start_pos)
        loss_end = criterion(end_logits, end_pos)
        
        # Combine the losses
        loss = loss_start + loss_end
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
    # Print epoch loss
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")