import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ModelHandler:
    
    def __init__(self, num_classes, device):
        self.device = device
        self.model = self._create_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.1)
    
    def _create_model(self, num_classes):
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.08),
            nn.Linear(64, num_classes)
        )
        
        for param in model.features.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss, correct = 0.0, 0
        
        for images, labels in tqdm(dataloader, desc="Training", unit="batch"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        
        epoch_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        
        return epoch_loss, accuracy

    def validate(self, dataloader):
        self.model.eval()
        epoch_loss, correct = 0.0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", unit="batch"):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                
                epoch_loss += loss.item() * images.size(0)
                correct += (output.argmax(1) == labels).sum().item()
            
        epoch_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        
        return epoch_loss, accuracy
    
    def train_model(self, train_loader, val_loader, num_epochs):
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('../models/best_model.pth')

        return history
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)