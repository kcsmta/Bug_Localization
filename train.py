from torch import nn, optim
import torch
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
# Freeze T5 encoder layers
for param in base_model.parameters():
    param.requires_grad = False
    
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({"additional_special_tokens": ["<t_cls>"]})

base_model.resize_token_embeddings(len(tokenizer))

hidden_size = base_model.config.d_model

model = BugLocalizationModel(base_model, tokenizer, hidden_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Hyper parameters:
batch_size = 16
num_epochs = 10
learning_rate = 5e-5
# learning_rate = 1e-5
# learning_rate = 1e-4

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
    correct_train_start, correct_train_end, total_train_samples = 0, 0, 0
    
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
        
        # Compute start and end token predictions
        pred_start = torch.argmax(start_logits, dim=1)
        pred_end = torch.argmax(end_logits, dim=1)
        
        # Calculate correct predictions for accuracy
        correct_train_start += (pred_start == start_pos).sum().item()
        correct_train_end += (pred_end == end_pos).sum().item()
        total_train_samples += start_pos.size(0)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_start_accuracy = correct_train_start / total_train_samples
    train_end_accuracy = correct_train_end / total_train_samples
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)  
    writer.add_scalar("Accuracy/Train_Start", train_start_accuracy, epoch + 1)
    writer.add_scalar("Accuracy/Train_End", train_end_accuracy, epoch + 1)
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    correct_val_start, correct_val_end, total_val_samples = 0, 0, 0
    
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
            
            # Compute predictions
            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)
            
            # Calculate correct predictions for accuracy
            correct_val_start += (pred_start == start_pos).sum().item()
            correct_val_end += (pred_end == end_pos).sum().item()
            total_val_samples += start_pos.size(0)
    
    # Calculate validation accuracy
    val_start_accuracy = correct_val_start / total_val_samples
    val_end_accuracy = correct_val_end / total_val_samples
    avg_val_loss = total_val_loss / len(valid_dataloader)
    
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation_Start", val_start_accuracy, epoch + 1)
    writer.add_scalar("Accuracy/Validation_End", val_end_accuracy, epoch + 1)
    
    # Print train and validation stats
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"Train Start Accuracy: {train_start_accuracy:.4f}, Train End Accuracy: {train_end_accuracy:.4f}")
    print(f"Validation Start Accuracy: {val_start_accuracy:.4f}, Validation End Accuracy: {val_end_accuracy:.4f}")


# Close the TensorBoard writer
writer.close()
print("Training and validation complete.")