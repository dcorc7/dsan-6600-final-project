import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import math

warnings.filterwarnings("ignore")

# dataset class
class CrashSequenceDataset(Dataset):
    def __init__(self, video_ids, labels_dict, base_path='annotated_frames'):
        self.base_path = base_path
        self.labels_dict = labels_dict
        self.video_ids = video_ids
        
        ## Calculate mean and std for normalization
        #all_data = []
        #for vid in video_ids:
        #    try:
        #        x = np.load(os.path.join(self.base_path, vid, 'frame_data.npy'))
        #        # Replace NaN/Inf values with zeros
        #        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        #        all_data.append(x)
        #    except Exception as e:
        #        print(f"Error loading video {vid}: {str(e)}")
        #        continue
        #        
        #all_data = np.concatenate(all_data)
        #self.mean = np.mean(all_data, axis=0)
        #self.std = np.std(all_data, axis=0) + 1e-6

        # DEBUG checks for labels
        for vid, label in labels_dict.items():
            if not isinstance(label, (int, float)) or np.isnan(label) or np.isinf(label):
                raise ValueError(f"Invalid label for video {vid}: {label}")
            if label not in [0, 1]:
                raise ValueError(f"Label for video {vid} must be 0 or 1, got {label}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        x = np.load(os.path.join(self.base_path, vid, 'frame_data.npy'))
        
        # total number of frames
        total_frames = len(x)
        # number of frames to keep (75%)
        frames_to_keep = int(total_frames * 0.75)
        
        # generate random indices while maintaining sequence order
        if frames_to_keep < total_frames:
            # create array of all indices and randomly select 75%
            all_indices = np.arange(total_frames)
            selected_indices = np.sort(np.random.choice(
                all_indices, 
                size=frames_to_keep, 
                replace=False
            ))
            x = x[selected_indices]
        
        # replace NaN/Inf values with zeros
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.labels_dict[vid]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

# Model class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerCrashClassifier(nn.Module):
    def __init__(self, input_dim=2*4 + 4 + 7 + 2 + 3 + 4, d_model=128, nhead=4, num_layers=2):

        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # transformer encoder layer with lower dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=512,  
            dropout=0.1,  # reduce dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm2 = nn.LayerNorm(d_model)
        
        # deeper classifier with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.norm1(x)
        
        # padding mask
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm2(x)
        
        # global average pooling instead of attention
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x.masked_fill(mask_expanded, 0.0)
        x = x.sum(dim=1) / lengths.unsqueeze(1).float()
        
        return self.classifier(x).squeeze(-1)

# training function
def train_one_fold(train_loader, val_loader, model, device, epochs=30):  # Increased epochs
    criterion = nn.BCEWithLogitsLoss()
    
    # optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4,
        total_steps=epochs * len(train_loader),
        pct_start=0.2,  # longer warmup
        div_factor=10,
        final_div_factor=10
    )
    
    clip_value = 1.0
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds, train_targets = [], []
        
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_preds.extend((outputs > 0).cpu().numpy())
            train_targets.extend(y.cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # validation phase
        model.eval()
        total_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                outputs = model(x, lengths)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                val_preds.extend((outputs > 0).cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # calculate and print metrics
        train_acc = sum(np.array(train_preds) == np.array(train_targets)) / len(train_targets)
        val_acc = sum(np.array(val_preds) == np.array(val_targets)) / len(val_targets)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_losses, val_losses, val_preds, val_targets

# main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_df = pd.read_csv('data/train_labels.csv')
    labels_df['id'] = labels_df['id'].astype(str).str.zfill(5)
    labels_dict = dict(zip(labels_df['id'], labels_df['target']))

    # create directory for plots if it doesn't exist
    os.makedirs('model_plots', exist_ok=True)

    all_ids = list(labels_dict.keys())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_ids, [labels_dict[i] for i in all_ids])):
        print(f"\n------ Fold {fold+1} ------")
        train_ids = [all_ids[i] for i in train_idx]
        val_ids = [all_ids[i] for i in val_idx]

        # verify class distribution in train and validation sets
        train_labels = [labels_dict[i] for i in train_ids]
        val_labels = [labels_dict[i] for i in val_ids]
        
        train_crash = sum(train_labels)
        train_no_crash = len(train_labels) - train_crash
        val_crash = sum(val_labels)
        val_no_crash = len(val_labels) - val_crash
        
        print(f"Train set - Crash: {train_crash}, No Crash: {train_no_crash}")
        print(f"Val set - Crash: {val_crash}, No Crash: {val_no_crash}")
        
        if train_crash == 0 or train_no_crash == 0:
            raise ValueError(f"Training set for fold {fold+1} is missing one class!")
        if val_crash == 0 or val_no_crash == 0:
            raise ValueError(f"Validation set for fold {fold+1} is missing one class!")

        train_ds = CrashSequenceDataset(train_ids, labels_dict)
        val_ds = CrashSequenceDataset(val_ids, labels_dict)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

        model = TransformerCrashClassifier().to(device)
        model, train_loss, val_loss, preds, targets = train_one_fold(train_loader, val_loader, model, device)

        # plot training vs validation loss
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'model_plots/fold_{fold+1}_loss_curve.png')
        plt.clf()

        # plot confusion matrix
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Crash', 'Crash'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix')
        plt.savefig(f'model_plots/fold_{fold+1}_confusion_matrix.png')
        plt.clf()

if __name__ == '__main__':
    main()

