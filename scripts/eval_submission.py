"""
Submission evaluation script for CUB-200 dataset or mini-ImageNet
Evaluates on validation set (shows accuracy) and generates test set predictions (CSV)
"""
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import pandas as pd

from ssl_vision.data_loader import SubmissionDataset, get_eval_transforms, submission_collate_fn
from ssl_vision.models import create_vision_transformer
from ssl_vision.utils import extract_features, KNNClassifier



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

                # Save best model
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_state_dict = classifier.state_dict().copy()

                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(linear_train_loader):.4f}, "
                      f"Val Acc: {accuracy*100:.2f}%")

    print(f"\nLinear Probing Best Validation Accuracy: {best_acc * 100:.2f}%")

    # Load best model
    classifier.load_state_dict(best_state_dict)

    return classifier, best_acc


def predict_with_classifier(classifier, features, device):
    """Generate predictions using a trained classifier"""
    classifier.eval()
    features_tensor = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        outputs = classifier(features_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()

    return predictions


def load_backbone(checkpoint_path: str, device: torch.device):
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

    return backbone


def save_predictions_to_csv(predictions, filenames, out_path):
    df = pd.DataFrame({
        'id': filenames,
        'class_id': predictions
    })
    df.to_csv(out_path, index=False)

    return df


def extract_experiment_id(checkpoint_dir: Path) -> str:
    """Extract experiment ID (long digit sequence) from checkpoint path."""
    for candidate in [checkpoint_dir] + list(checkpoint_dir.parents):
        name = candidate.name
        match = re.search(r'(\d{5,})', name)
        if match:
            return match.group(1)
    raise ValueError(f"Could not infer experiment ID from path: {checkpoint_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="submission")
def main(cfg: DictConfig):
    eval_cfg = cfg.evaluation
    assert len(list(eval_cfg.data.keys())) == 1, "Only one dataset is supported for evaluation for now"

    """Main evaluation entry point"""
    dataset_name = eval_cfg.data['0'].dataset_name
    print(f"Dataset name: {dataset_name}")
    print(f"{dataset_name} Submission Evaluation")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if not hasattr(cfg, 'evaluation'):
        print("Error: No evaluation config found. Add evaluation settings to config.")
        return

    eval_cfg = cfg.evaluation
    method = eval_cfg.get('method', 'linear').lower()
    if method not in {'knn', 'linear'}:
        print(f"Error: Unknown evaluation method '{method}'. Choose 'knn' or 'linear'.")
        return
    method_label = 'k-NN' if method == 'knn' else 'Linear Probing'

    checkpoint_dir = Path(eval_cfg.checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return

    checkpoint_paths = sorted((p for p in checkpoint_dir.iterdir()), reverse=True)
    if not checkpoint_paths:
        print(f"Error: No checkpoint files found in {checkpoint_dir}")
        return

    experiment_id = extract_experiment_id(checkpoint_dir)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"\nFound {len(checkpoint_paths)} checkpoints in: {checkpoint_dir}")
    print(f"Evaluation method: {method_label}")
    print(f"\nLoading {dataset_name} dataset from: {eval_cfg.data['0'].data_dir}")

    # Evaluation transforms
    eval_transform = get_eval_transforms()

    # Load train split
    train_dataset = SubmissionDataset(
        root_dir=eval_cfg.data['0'].data_dir,
        split='train',
        transform=eval_transform,
    )

    # Load validation split
    val_dataset = SubmissionDataset(
        root_dir=eval_cfg.data['0'].data_dir,
        split='val',
        transform=eval_transform,
    )

    # Load test split
    test_dataset = SubmissionDataset(
        root_dir=eval_cfg.data['0'].data_dir,
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

    # Setup results directory
    results_dir = Path(eval_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find existing directories with numbered suffixes
    pattern = re.compile(rf"^{re.escape(experiment_id)}_(\d+)$")
    max_num = 0

    for item in results_dir.iterdir():
        match = pattern.match(item.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    # Create new directory with incremented number
    new_num = max_num + 1
    experiment_results_dir = results_dir / f"{experiment_id}_{new_num}"
    print(f"Creating new results directory: {experiment_results_dir} for results.")

    experiment_results_dir.mkdir(parents=True, exist_ok=True)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print("\n" + "="*80)
    print("BEGINNING CHECKPOINT EVALUATION")
    print("="*80)

    results = []
    feature_dim = None
    num_classes = None

    knn_k = eval_cfg.get('knn_k', 20)
    probe_epochs = eval_cfg.get('linear_probe_epochs', 100)
    probe_lr = eval_cfg.get('linear_probe_lr', 0.001)
    linear_batch = eval_cfg.get('batch_size', 256)

    for checkpoint_path in checkpoint_paths:
        print("\n" + "-"*80)
        print(f"Processing checkpoint: {checkpoint_path}")
        print("-"*80)

        backbone = load_backbone(str(checkpoint_path), device)
        backbone.eval()

        print("Extracting train features...")
        train_features, train_labels, _ = extract_features(backbone, train_loader, device)

        print("Extracting validation features...")
        val_features, val_labels, _ = extract_features(backbone, val_loader, device)

        test_features = None
        test_filenames = None
        if method == 'linear':
            print("Extracting test features...")
            test_features, _, test_filenames = extract_features(backbone, test_loader, device)

        if feature_dim is None:
            feature_dim = train_features.shape[1]
        if num_classes is None:
            num_classes = len(np.unique(train_labels))

        if method == 'knn':
            knn_classifier = KNNClassifier(k=knn_k)
            knn_classifier.train(train_features, train_labels)
            val_acc = knn_classifier.evaluate(val_features, val_labels)
            print(f"k-NN Validation Accuracy: {val_acc * 100:.2f}%")
            pred_path = None
        elif method == 'linear':
            linear_classifier, val_acc = linear_probe_training(
                train_features,
                train_labels,
                val_features,
                val_labels,
                num_classes=num_classes,
                device=device,
                epochs=probe_epochs,
                lr=probe_lr,
                batch_size=linear_batch,
            )
            print("Generating predictions on test set using best linear model...")
            test_linear_pred = predict_with_classifier(linear_classifier, test_features, device)
            pred_path = experiment_results_dir / f"{dataset_name}_{checkpoint_path.stem}.csv"
            save_predictions_to_csv(test_linear_pred, test_filenames, pred_path)
            print(f"✓ Linear probing predictions saved to: {pred_path}")
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

        results.append({
            'checkpoint': str(checkpoint_path),
            'accuracy': val_acc,
            'method_label': method_label,
            'prediction_path': str(pred_path) if pred_path else None,
        })

        del backbone
        torch.cuda.empty_cache()

    if not results:
        print("No checkpoints were evaluated.")
        return

    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for idx, res in enumerate(sorted_results, 1):
        print(f"{idx}. {res['checkpoint']} - {res['method_label']} Accuracy: {res['accuracy'] * 100:.2f}%")
        if res['prediction_path']:
            print(f"   Predictions: {res['prediction_path']}")

    summary_filename = f"{eval_cfg.data['0'].dataset_name}_{method.lower()}_results.txt"
    summary_file = experiment_results_dir / summary_filename
    with open(summary_file, 'w') as f:
        f.write(f"{eval_cfg.data['0'].dataset_name} Submission Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint directory: {checkpoint_dir}\n")
        f.write(f"Evaluation method: {method_label}\n")
        f.write(f"Checkpoint count: {len(checkpoint_paths)}\n")
        f.write(f"Dataset: {eval_cfg.data['0'].data_dir}\n")
        if num_classes is not None:
            f.write(f"Number of classes: {num_classes}\n")
        if feature_dim is not None:
            f.write(f"Feature dimension: {feature_dim}\n")
        f.write("\nDataset sizes:\n")
        f.write(f"  Train: {train_size} samples\n")
        f.write(f"  Val:   {val_size} samples\n")
        f.write(f"  Test:  {test_size} samples\n")
        f.write("\nHyperparameters:\n")
        if method == 'knn':
            f.write(f"  k-NN k: {knn_k}\n")
        else:
            f.write(f"  Linear probe epochs: {probe_epochs}\n")
            f.write(f"  Linear probe lr: {probe_lr}\n")
            f.write(f"  Linear probe batch size: {linear_batch}\n")
        f.write("\nResults (sorted by accuracy):\n")
        for res in sorted_results:
            f.write(f"  {res['checkpoint']}\n")
            f.write(f"    Accuracy: {res['accuracy'] * 100:.2f}%\n")
            if res['prediction_path']:
                f.write(f"    Predictions: {res['prediction_path']}\n")

    print(f"\n✓ Summary saved to: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
