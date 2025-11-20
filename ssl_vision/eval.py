"""
Evaluation script for self-supervised models
Supports linear probing and k-NN evaluation
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from data_loader import HuggingFaceImageDataset
from models import create_vision_transformer
from torchvision import transforms


def get_eval_transforms():
    """Simple transforms for evaluation (no augmentation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features from a pretrained model"""
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)

        # Forward pass through backbone
        features = model(images)

        # Use [CLS] token
        cls_features = features[:, 0]  # Shape: [B, D]

        all_features.append(cls_features.cpu())
        all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels


def knn_evaluation(train_features, train_labels, test_features, test_labels, k=20):
    """k-NN evaluation"""
    print(f"\n{'='*60}")
    print(f"k-NN Evaluation (k={k})")
    print(f"{'='*60}")

    # Normalize features
    train_features = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
    test_features = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    # Train k-NN classifier
    print(f"Training k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)

    # Predict
    print(f"Making predictions...")
    predictions = knn.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Get top-5 accuracy if applicable
    if len(np.unique(train_labels)) > 5:
        knn_proba = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn_proba.fit(train_features, train_labels)
        proba = knn_proba.predict_proba(test_features)
        top5_acc = top_k_accuracy_score(test_labels, proba, k=5)
    else:
        top5_acc = None

    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {accuracy * 100:.2f}%")
    if top5_acc is not None:
        print(f"  Top-5 Accuracy: {top5_acc * 100:.2f}%")

    return accuracy, top5_acc


def linear_probe_evaluation(
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_classes,
    device,
    epochs=100,
    lr=0.001,
    batch_size=256,
):
    """Linear probing evaluation"""
    print(f"\n{'='*60}")
    print(f"Linear Probing Evaluation")
    print(f"{'='*60}")

    # Create linear classifier
    input_dim = train_features.shape[1]
    classifier = nn.Linear(input_dim, num_classes).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)

    # Convert to tensors
    train_features_tensor = torch.from_numpy(train_features).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_features_tensor = torch.from_numpy(test_features).float().to(device)
    test_labels_tensor = torch.from_numpy(test_labels).long().to(device)

    # Create dataset and loader
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"Training linear classifier for {epochs} epochs...")
    best_acc = 0

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            classifier.eval()
            with torch.no_grad():
                test_outputs = classifier(test_features_tensor)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == test_labels_tensor).float().mean().item()

                # Top-5 accuracy
                if num_classes > 5:
                    _, top5_pred = test_outputs.topk(5, 1, True, True)
                    top5_correct = top5_pred.eq(test_labels_tensor.view(-1, 1).expand_as(top5_pred))
                    top5_acc = top5_correct.any(dim=1).float().mean().item()
                else:
                    top5_acc = None

                if accuracy > best_acc:
                    best_acc = accuracy

                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, "
                      f"Acc: {accuracy*100:.2f}%", end="")
                if top5_acc is not None:
                    print(f", Top-5: {top5_acc*100:.2f}%")
                else:
                    print()

    print(f"\nBest Accuracy: {best_acc * 100:.2f}%")
    return best_acc, top5_acc if num_classes > 5 else None


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation entry point"""
    print("=" * 80)
    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check for required evaluation config
    if not hasattr(cfg, 'evaluation'):
        print("Error: No evaluation config found. Add evaluation settings to config.")
        return

    # Load checkpoint
    checkpoint_path = cfg.evaluation.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    print("Creating model...")
    backbone = create_vision_transformer(cfg)

    # Load weights (handle both full model and backbone-only checkpoints)
    if 'student_state_dict' in checkpoint:
        # Full training checkpoint
        state_dict = checkpoint['student_state_dict']
        # Extract backbone weights
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                backbone_state_dict[k.replace('backbone.', '')] = v
        if backbone_state_dict:
            backbone.load_state_dict(backbone_state_dict)
        else:
            backbone.load_state_dict(state_dict)
    else:
        backbone.load_state_dict(checkpoint)

    backbone = backbone.to(device)
    backbone.eval()

    print(f"Model loaded successfully!")

    # Evaluation dataset configuration:
    # Use the dataset specified in cfg.data.dataset_name (e.g., configs/data/cifar100.yaml)
    print(f"\nLoading evaluation dataset from data config: {cfg.data.dataset_name}")
    print(f"  train split: {cfg.data.train_split}")
    print(f"  test  split: {cfg.data.val_split}")
    print(f"  image key : {cfg.data.image_key}")

    # Evaluation transforms (no augmentation, keep original image size)
    eval_transform = get_eval_transforms()

    # Load train split for feature extraction
    train_dataset = HuggingFaceImageDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        transform=eval_transform,
        cache_dir=cfg.data.cache_dir,
        streaming=cfg.data.streaming,
        image_key=cfg.data.image_key,
    )

    # Load test split
    test_dataset = HuggingFaceImageDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.val_split,
        transform=eval_transform,
        cache_dir=cfg.data.cache_dir,
        streaming=cfg.data.streaming,
        image_key=cfg.data.image_key,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.evaluation.get('batch_size', 256),
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.get('batch_size', 256),
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Extract features
    print("\nExtracting features from train set...")
    train_features, train_labels = extract_features(backbone, train_loader, device)

    print("Extracting features from test set...")
    test_features, test_labels = extract_features(backbone, test_loader, device)

    num_classes = len(np.unique(train_labels))
    print(f"\nNumber of classes: {num_classes}")
    print(f"Feature dimension: {train_features.shape[1]}")

    # Run evaluations
    results = {}

    # k-NN evaluation

    knn_k = cfg.evaluation.get('knn_k', 20)
    knn_top1, knn_top5 = knn_evaluation(
        train_features, train_labels,
        test_features, test_labels,
        k=knn_k
    )
    results['knn_top1'] = knn_top1
    if knn_top5 is not None:
        results['knn_top5'] = knn_top5

    # Linear probing evaluation
    linear_top1, linear_top5 = linear_probe_evaluation(
        train_features, train_labels,
        test_features, test_labels,
        num_classes=num_classes,
        device=device,
        epochs=cfg.evaluation.get('linear_probe_epochs', 100),
        lr=cfg.evaluation.get('linear_probe_lr', 0.001),
        batch_size=cfg.evaluation.get('batch_size', 256),
    )
    results['linear_top1'] = linear_top1
    if linear_top5 is not None:
        results['linear_top5'] = linear_top5

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    for key, value in results.items():
        print(f"{key}: {value*100:.2f}%")
    print(f"{'='*80}")

    # Save results
    results_dir = Path(cfg.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"results_{cfg.experiment_name}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Eval Dataset: {cfg.data.dataset_name}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"\nResults:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value*100:.2f}%\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()


