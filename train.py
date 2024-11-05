import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.tensorboard import SummaryWriter
from models import BugLocalizationModel
from dataset import BugLocalizationDataset

# File paths and model setup
training_file = './data/train_tufano_small.csv'
validation_file = './data/valid_tufano_small.csv'
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

# Hyper parameters:
batch_size = 4
num_epochs = 10
learning_rate = 5e-5

# Initialize the dataset and DataLoader
train_dataset = BugLocalizationDataset(csv_file=training_file, tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = BugLocalizationDataset(csv_file=validation_file, tokenizer=tokenizer)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# TensorBoard setup
writer = SummaryWriter(log_dir="./logs")

# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0
    
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        start_pos = batch["start_pos"].to(device)
        end_pos = batch["end_pos"].to(device)
        
        # Forward pass
        start_logits, end_logits = model(input_ids=input_ids)
        
        # Loss computation
        loss_start = criterion(start_logits, start_pos)
        loss_end = criterion(end_logits, end_pos)
        loss = loss_start + loss_end
        total_train_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)  # Log train loss
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    
    with torch.no_grad():  # No gradient calculation for validation
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(device)
            start_pos = batch["start_pos"].to(device)
            end_pos = batch["end_pos"].to(device)
            
            # Forward pass
            start_logits, end_logits = model(input_ids=input_ids)
            
            # Loss computation
            loss_start = criterion(start_logits, start_pos)
            loss_end = criterion(end_logits, end_pos)
            loss = loss_start + loss_end
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(valid_dataloader)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)  # Log validation loss
    
    # Print train and validation loss for each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Close the TensorBoard writer
writer.close()
print("Training and validation complete.")