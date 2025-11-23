"""
Submission evaluation script for CUB-200 dataset
Evaluates on validation set (shows accuracy) and generates test set predictions (CSV)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import pandas as pd

from ssl_vision.data_loader import SubmissionCUBDataset, get_eval_transforms
from ssl_vision.models import create_vision_transformer


def submission_collate_fn(batch):
    """Custom collate function for SubmissionCUBDataset that handles filenames"""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return images, labels, filenames


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features from a pretrained model along with filenames"""
    model.eval()
    all_features = []
    all_labels = []
    all_filenames = []

    for images, labels, filenames in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)

        # Forward pass through backbone
        features = model(images)

        # Use [CLS] token
        cls_features = features[:, 0]  # Shape: [B, D]

        all_features.append(cls_features.cpu())
        all_labels.append(labels.cpu())
        all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels, all_filenames


def knn_evaluation(train_features, train_labels, test_features, test_labels, k=20, is_test_split=False):
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

    # Calculate accuracy only if we have real labels (validation set)
    if not is_test_split:
        accuracy = accuracy_score(test_labels, predictions)
        num_classes = len(np.unique(train_labels))

        # Get top-5 accuracy if applicable
        top5_acc = None
        if num_classes > 5:
            proba = knn.predict_proba(test_features)
            top5_acc = top_k_accuracy_score(test_labels, proba, k=5)

        print(f"\nValidation Results:")
        print(f"  Top-1 Accuracy: {accuracy * 100:.2f}%")
        if top5_acc is not None:
            print(f"  Top-5 Accuracy: {top5_acc * 100:.2f}%")

        return predictions, accuracy, top5_acc
    else:
        print(f"Test set - predictions generated (no labels available)")
        return predictions, None, None


def linear_probe_training(
    train_features,
    train_labels,
    val_features,
    val_labels,
    num_classes,
    device,
    epochs=100,
    lr=0.001,
    batch_size=256,
):
    """Train linear classifier and return the best model"""
    print(f"\n{'='*60}")
    print(f"Linear Probing Training")
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
    val_features_tensor = torch.from_numpy(val_features).float().to(device)
    val_labels_tensor = torch.from_numpy(val_labels).long().to(device)

    # Create dataset and loader
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)
    linear_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"Training linear classifier for {epochs} epochs...")
    best_acc = 0
    best_top5_acc = None
    best_state_dict = None

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for features, labels in linear_train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            classifier.eval()
            with torch.no_grad():
                val_outputs = classifier(val_features_tensor)
                _, predicted = torch.max(val_outputs, 1)

                accuracy = (predicted == val_labels_tensor).float().mean().item()

                # Top-5 accuracy
                if num_classes > 5:
                    _, top5_pred = val_outputs.topk(5, 1, True, True)
                    top5_correct = top5_pred.eq(val_labels_tensor.view(-1, 1).expand_as(top5_pred))
                    top5_acc = top5_correct.any(dim=1).float().mean().item()
                else:
                    top5_acc = None

                # Save best model
                if accuracy > best_acc:
                    best_acc = accuracy
                    if top5_acc is not None:
                        best_top5_acc = top5_acc
                    best_state_dict = classifier.state_dict().copy()

                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(linear_train_loader):.4f}, "
                      f"Val Acc: {accuracy*100:.2f}%", end="")
                if top5_acc is not None:
                    print(f", Val Top-5: {top5_acc*100:.2f}%")
                else:
                    print()

    print(f"\nBest Validation Accuracy: {best_acc * 100:.2f}%")
    if best_top5_acc is not None:
        print(f"Best Validation Top-5 Accuracy: {best_top5_acc * 100:.2f}%")

    # Load best model
    classifier.load_state_dict(best_state_dict)

    return classifier, best_acc, best_top5_acc


def predict_with_classifier(classifier, features, device):
    """Generate predictions using a trained classifier"""
    classifier.eval()
    features_tensor = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        outputs = classifier(features_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()

    return predictions


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation entry point"""
    print("=" * 80)
    print("CUB-200 Submission Evaluation")
    print("=" * 80)
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
    snapped_cfg = OmegaConf.create(checkpoint['config'])

    # Create model
    print("Creating model...")
    backbone = create_vision_transformer(snapped_cfg)

    # Load weights
    if 'student_state_dict' in checkpoint:
        state_dict = checkpoint['student_state_dict']
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if 'backbone.' in k:
                new_key = k.split('backbone.', 1)[1]
                backbone_state_dict[new_key] = v
        if backbone_state_dict:
            backbone.load_state_dict(backbone_state_dict)
        else:
            backbone.load_state_dict(state_dict)
    else:
        backbone.load_state_dict(checkpoint)

    backbone = backbone.to(device)
    backbone.eval()
    print(f"Model loaded successfully!")

    # Check dataset configuration
    if cfg.data.get('dataset_type') != 'submission_cub':
        print("Error: This script requires dataset_type='submission_cub'")
        print("Please use configs/data/submission_cub200.yaml")
        return

    print(f"\nLoading CUB-200 dataset from: {cfg.data.data_dir}")

    # Evaluation transforms
    eval_transform = get_eval_transforms()

    # Load train split
    train_dataset = SubmissionCUBDataset(
        root_dir=cfg.data.data_dir,
        split='train',
        transform=eval_transform,
    )

    # Load validation split
    val_dataset = SubmissionCUBDataset(
        root_dir=cfg.data.data_dir,
        split='val',
        transform=eval_transform,
    )

    # Load test split
    test_dataset = SubmissionCUBDataset(
        root_dir=cfg.data.data_dir,
        split='test',
        transform=eval_transform,
    )

    batch_size = cfg.evaluation.get('batch_size', 256)
    num_workers = cfg.training.num_workers

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=submission_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=submission_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=submission_collate_fn,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Extract features from all splits
    print("\n" + "="*80)
    print("EXTRACTING FEATURES")
    print("="*80)

    print("\nExtracting features from train set...")
    train_features, train_labels, _ = extract_features(backbone, train_loader, device)

    print("Extracting features from validation set...")
    val_features, val_labels, _ = extract_features(backbone, val_loader, device)

    print("Extracting features from test set...")
    test_features, test_labels, test_filenames = extract_features(backbone, test_loader, device)

    num_classes = len(np.unique(train_labels))
    print(f"\nNumber of classes: {num_classes}")
    print(f"Feature dimension: {train_features.shape[1]}")

    # Setup results directory
    results_dir = Path(cfg.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # STEP 1: k-NN EVALUATION
    # =======================================================================
    print("\n" + "="*80)
    print("STEP 1: k-NN EVALUATION")
    print("="*80)

    knn_k = cfg.evaluation.get('knn_k', 20)

    # Train k-NN on train set, evaluate on validation set
    val_knn_pred, val_knn_acc, val_knn_top5 = knn_evaluation(
        train_features, train_labels,
        val_features, val_labels,
        k=knn_k,
        is_test_split=False,
    )

    # Generate k-NN predictions on test set (same classifier)
    print(f"\nGenerating k-NN predictions on test set...")
    test_knn_pred, _, _ = knn_evaluation(
        train_features, train_labels,
        test_features, test_labels,
        k=knn_k,
        is_test_split=True,
    )

    # Save k-NN predictions
    knn_csv_path = results_dir / f"knn_predictions_{cfg.experiment_name}.csv"
    df_knn = pd.DataFrame({
        'id': test_filenames,
        'class_id': test_knn_pred
    })
    df_knn.to_csv(knn_csv_path, index=False)
    print(f"✓ k-NN predictions saved to: {knn_csv_path}")

    # =======================================================================
    # STEP 2: LINEAR PROBING
    # =======================================================================
    print("\n" + "="*80)
    print("STEP 2: LINEAR PROBING")
    print("="*80)

    # Train linear classifier on train set, evaluate on validation set
    # Keep the best model based on validation accuracy
    linear_classifier, val_linear_acc, val_linear_top5 = linear_probe_training(
        train_features, train_labels,
        val_features, val_labels,
        num_classes=num_classes,
        device=device,
        epochs=cfg.evaluation.get('linear_probe_epochs', 100),
        lr=cfg.evaluation.get('linear_probe_lr', 0.001),
        batch_size=cfg.evaluation.get('batch_size', 256),
    )

    # Use the best model to predict on test set (no retraining!)
    print(f"\nGenerating predictions on test set using best model...")
    test_linear_pred = predict_with_classifier(linear_classifier, test_features, device)

    # Save linear predictions
    linear_csv_path = results_dir / f"linear_predictions_{cfg.experiment_name}.csv"
    df_linear = pd.DataFrame({
        'id': test_filenames,
        'class_id': test_linear_pred
    })
    df_linear.to_csv(linear_csv_path, index=False)
    print(f"✓ Linear probing predictions saved to: {linear_csv_path}")

    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nValidation Set Results:")
    print(f"  k-NN (k={knn_k}):")
    print(f"    Top-1 Accuracy: {val_knn_acc * 100:.2f}%")
    if val_knn_top5 is not None:
        print(f"    Top-5 Accuracy: {val_knn_top5 * 100:.2f}%")

    print(f"\n  Linear Probing (best model):")
    print(f"    Top-1 Accuracy: {val_linear_acc * 100:.2f}%")
    if val_linear_top5 is not None:
        print(f"    Top-5 Accuracy: {val_linear_top5 * 100:.2f}%")

    print(f"\nTest Set Predictions (saved as CSV):")
    print(f"  k-NN predictions:    {knn_csv_path}")
    print(f"  Linear predictions:  {linear_csv_path}")

    print(f"\n💡 Recommendation: Use linear probing predictions for submission")
    print(f"   (typically achieves higher accuracy than k-NN)")

    # Save summary to file
    summary_file = results_dir / f"summary_{cfg.experiment_name}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"CUB-200 Submission Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {cfg.data.data_dir}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Feature dimension: {train_features.shape[1]}\n")
        f.write(f"\nDataset sizes:\n")
        f.write(f"  Train: {len(train_features)} samples\n")
        f.write(f"  Val:   {len(val_features)} samples\n")
        f.write(f"  Test:  {len(test_features)} samples\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  k-NN k: {knn_k}\n")
        f.write(f"  Linear probe epochs: {cfg.evaluation.get('linear_probe_epochs', 100)}\n")
        f.write(f"  Linear probe lr: {cfg.evaluation.get('linear_probe_lr', 0.001)}\n")
        f.write(f"\nValidation Set Results:\n")
        f.write(f"  k-NN Top-1 Accuracy: {val_knn_acc * 100:.2f}%\n")
        if val_knn_top5 is not None:
            f.write(f"  k-NN Top-5 Accuracy: {val_knn_top5 * 100:.2f}%\n")
        f.write(f"  Linear Top-1 Accuracy: {val_linear_acc * 100:.2f}%\n")
        if val_linear_top5 is not None:
            f.write(f"  Linear Top-5 Accuracy: {val_linear_top5 * 100:.2f}%\n")
        f.write(f"\nTest Set Predictions:\n")
        f.write(f"  k-NN CSV: {knn_csv_path}\n")
        f.write(f"  Linear CSV: {linear_csv_path}\n")
        f.write(f"\nRecommendation: Use linear probing predictions for submission.\n")

    print(f"\n✓ Summary saved to: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
