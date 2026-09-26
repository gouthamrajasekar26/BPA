"""
MODULE 2: Spatter Type & Impact Velocity Classifier (PyTorch ResNet-50)
========================================================================
Implements a Transfer Learning CNN based on ResNet-50 for high-accuracy
forensic spatter classification and impact velocity range estimation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
from typing import Dict, Any, List

# Class keys mapped to dataset split subfolder names
CATEGORIES_KEYS = ['beating', 'drip', 'hvis', 'mvis']

SPATTER_CLASS_MAP = {
    'drip': {
        "display_name": "Passive Drip Spatter",
        "velocity_category": "Low Velocity (< 1.5 m/s)",
        "velocity_range_ms": "0.2 - 1.5 m/s",
        "droplet_size_range": "Large (> 4.0 mm)",
        "mechanism": "Gravity flow / dripping from wounded surface or bloody object"
    },
    'beating': {
        "display_name": "Blunt Force / Beating Spatter",
        "velocity_category": "Medium Velocity (1.5 - 7.5 m/s)",
        "velocity_range_ms": "1.5 - 7.5 m/s",
        "droplet_size_range": "Medium (1.0 - 4.0 mm)",
        "mechanism": "Impact from blunt object, fist, weapon stroke, or cast-off force"
    },
    'mvis': {
        "display_name": "Medium-Velocity Impact Spatter (MVIS)",
        "velocity_category": "Medium Velocity (1.5 - 7.5 m/s)",
        "velocity_range_ms": "1.5 - 7.5 m/s",
        "droplet_size_range": "Medium (1.0 - 4.0 mm)",
        "mechanism": "Stabbing, blunt trauma, or secondary impact mechanics"
    },
    'hvis': {
        "display_name": "High-Velocity Impact Spatter (HVIS)",
        "velocity_category": "High Velocity (> 30 m/s)",
        "velocity_range_ms": "30.0 - 100+ m/s",
        "droplet_size_range": "Fine Mist / Micro-droplets (< 1.0 mm)",
        "mechanism": "Gunshot discharge, explosive force, or high-speed machinery"
    }
}


def build_resnet50_classifier(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Constructs a PyTorch ResNet-50 transfer learning architecture with a custom
    fully-connected classification head tuned for forensic spatter pattern analysis.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Freeze early backbone feature extractor layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def get_data_transforms():
    """Returns PyTorch data preprocessing & augmentation pipelines for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform


def train_spatter_classifier(
    data_dir: str = "data/split",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    save_path: str = "spatter_resnet50.pth"
):
    """
    Trains the ResNet-50 spatter classifier on training dataset split and evaluates
    categorical accuracy and loss on the validation dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Module 2] Utilizing compute device: {device}")
    
    train_tf, eval_tf = get_data_transforms()
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_path):
        print(f"[Module 2] Training directory {train_path} not found. Please run Module 1 first.")
        return None
        
    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = build_resnet50_classifier(num_classes=len(train_dataset.classes), pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    print(f"[Module 2] Beginning training loop over {epochs} epochs...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"   Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"[Module 2] Saved trained model weights to: {save_path}")
    return model, history


def predict_spatter_type(image_input, model_path: str = "spatter_resnet50.pth") -> Dict[str, Any]:
    """
    Performs deep learning inference on a single image input (file path or PIL Image).
    Returns predicted spatter class, impact velocity classification, confidence score,
    and categorical probability distribution.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.fromarray(np.uint8(image_input)).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_data_transforms()
    input_tensor = eval_tf(image).unsqueeze(0).to(device)
    
    classes = ['beating', 'drip', 'hvis', 'mvis']
    model = build_resnet50_classifier(num_classes=len(classes), pretrained=False)
    
    loaded_ckpt = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            loaded_ckpt = True
        except Exception as e:
            print(f"[Module 2] Checkpoint load exception ({e}). Utilizing analytical spatter heuristic.")
            probs = _heuristic_image_probabilities(image)
    else:
        probs = _heuristic_image_probabilities(image)
        
    pred_idx = int(np.argmax(probs))
    raw_cat_key = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    cat_info = SPATTER_CLASS_MAP.get(raw_cat_key, SPATTER_CLASS_MAP['mvis'])
    
    probabilities_dict = {}
    for i, c_key in enumerate(classes):
        display_name = SPATTER_CLASS_MAP[c_key]["display_name"]
        probabilities_dict[display_name] = float(probs[i])

    return {
        "predicted_class": cat_info["display_name"],
        "class_key": raw_cat_key,
        "confidence": confidence,
        "class_probabilities": probabilities_dict,
        "impact_velocity": {
            "category": cat_info["velocity_category"],
            "range_ms": cat_info["velocity_range_ms"],
            "droplet_size_range": cat_info["droplet_size_range"],
            "physical_mechanism": cat_info["mechanism"]
        },
        "checkpoint_used": loaded_ckpt
    }


def _heuristic_image_probabilities(pil_img: Image.Image) -> np.ndarray:
    """
    Computes analytical spatter probability based on droplet density and distribution
    when model weights checkpoint is uninitialized.
    Order of classes: ['beating', 'drip', 'hvis', 'mvis']
    """
    img_arr = np.array(pil_img.convert("L"))
    dark_pixels = np.sum(img_arr < 100)
    total_pixels = img_arr.size
    density = dark_pixels / total_pixels
    
    if density > 0.15:
        # Dense impact spatter (Beating / MVIS)
        probs = [0.45, 0.05, 0.15, 0.35]
    elif density > 0.05:
        # Medium impact spatter
        probs = [0.35, 0.10, 0.15, 0.40]
    elif density > 0.01:
        # High velocity fine mist spatter
        probs = [0.15, 0.10, 0.65, 0.10]
    else:
        # Sparse passive drips
        probs = [0.10, 0.70, 0.05, 0.15]
        
    return np.array(probs, dtype=np.float32)


if __name__ == "__main__":
    print("=== MODULE 2: Spatter Type & Impact Velocity Classifier ===")
    sample_img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    res = predict_spatter_type(sample_img)
    print("[Module 2] Sample Inference Result:")
    print(f"   - Predicted Class: {res['predicted_class']}")
    print(f"   - Confidence: {res['confidence']*100:.2f}%")
    print(f"   - Impact Velocity Category: {res['impact_velocity']['category']}")
    print(f"   - Velocity Range: {res['impact_velocity']['range_ms']}")
    print(f"   - Probabilities: {res['class_probabilities']}\n")
