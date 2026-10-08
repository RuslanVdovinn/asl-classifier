from torchvision import transforms

def get_transforms(model_type="simple_cnn", img_size=200, crop_padding=20, mean=0.5, std=0.5):
    
    if model_type == "simple_cnn":
        train_t = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 1 channel
            transforms.Resize((img_size + crop_padding, img_size + crop_padding)),
            transforms.RandomAffine(15, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=127),
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        val_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    
    elif model_type in ("resnet", "efficientnet"):
        # RGB, 3 канала, ImageNet нормализация
        train_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        val_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return train_t, val_t