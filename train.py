import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_data_loaders
from src.model import EuroSATCNN

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting computing engine: {device}")
    
    print("Streaming dataset arrays from TFDS...")
    train_loader, val_loader, class_names = get_data_loaders(config)
    print(f"Successfully loaded classes: {class_names}")
    
    model = EuroSATCNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] -> "
              f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'config': config
            }, checkpoint_path)
            print(f" Saved structural checkpoint to: {checkpoint_path}")

if __name__ == '__main__':
    main()