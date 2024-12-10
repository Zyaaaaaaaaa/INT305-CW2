import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # 使用论文中的卷积设置
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                            padding=kernel_size//2, bias=False)  # 论文使用无偏置
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)  # 论文使用SiLU(Swish)激活函数
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),  # 改回ReLU
            nn.Dropout(0.1),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.drop_path = nn.Dropout(0.05)  # 降低dropout率
        
    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x.transpose(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MobileViT(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # 论文中的通道数设置
        self.conv1 = ConvLayer(in_channels, 16, kernel_size=3, stride=2)
        
        self.conv2 = ConvLayer(16, 32, kernel_size=3)
        self.transformer1 = TransformerBlock(dim=32)
        
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.transformer2 = TransformerBlock(dim=64)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)  # 论文中的dropout率
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer1(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.conv3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer2(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Training configuration
def get_training_config():
    # 优化后的配置
    config = {
        'lr': 0.001,  # 降低学习率
        'weight_decay': 0.01,  # 保持不变
        'epochs': 20,  # 保持不变
        'batch_size': 128,  # 改回较小的batch size
        'optimizer': 'AdamW',  # 保持使用AdamW
        'scheduler': 'CosineAnnealingLR'  # 保持使用余弦退火
    }
    return config

def train_model(model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': 0.0
    }
    
    model = model.to(device)
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}', ncols=80)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{train_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Testing phase
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Update best accuracy
        if test_acc > history['best_acc']:
            history['best_acc'] = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    return history

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(dataloader), 100. * correct / total

def plot_training_history(history):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['test_loss'], label='Test Loss')
        ax1.set_title('Loss vs. Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['test_acc'], label='Test Accuracy')
        ax2.set_title('Accuracy vs. Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot training history due to: {e}")
        print("Training history data:", history)

# Usage example
if __name__ == '__main__':
    try:
        # Initialize model and configuration
        model = MobileViT(in_channels=1, num_classes=10)
        config = get_training_config()
        
        # Train model
        history = train_model(model, config)
        
        # Plot training history
        plot_training_history(history)
        
        print(f"Best Test Accuracy: {history['best_acc']:.2f}%")
    except Exception as e:
        print(f"Error during training: {e}")