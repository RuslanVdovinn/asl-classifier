import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Блок 1: (B, 1, 200, 200) → (B, 32, 100, 100)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 2: (B, 32, 100, 100) → (B, 64, 50, 50)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 3: (B, 64, 50, 50) → (B, 128, 25, 25)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1)),     # [B, 128, 1, 1]
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, 128, 1, 1) → (B, 128)
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),     # (B, 128) → (B, 29)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN(num_classes=29)
    print(model)
    
    # Число параметров
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего параметров: {total:,}")
    print(f"Обучаемых: {trainable:,}")
    
    # Прямой проход
    x = torch.randn(2, 1, 200, 200)
    y = model(x)
    print(f"\nВход:  {x.shape}")
    print(f"Выход: {y.shape}")   # (2, 29)