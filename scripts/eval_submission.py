"""
Submission evaluation script for CUB-200 dataset or mini-ImageNet
Evaluates on validation set (shows accuracy) and generates test set predictions (CSV)
"""
import re
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import pandas as pd

from ssl_vision.data_loader import SubmissionDataset, get_eval_transforms, submission_collate_fn
from ssl_vision.models import create_vision_transformer
from ssl_vision.utils import extract_features, KNNClassifier


METHOD_LABELS = {"linear": "Linear Probing", "knn": "k-NN"}
SUPPORTED_METHODS = set(METHOD_LABELS.keys())
SPLIT_ORDER = ("train", "val", "test")



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
    print("Linear Probing Training")
    print(f"{'='*60}")

    input_dim = train_features.shape[1]
    classifier = nn.Linear(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)

    train_features_tensor = torch.from_numpy(train_features).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    val_features_tensor = torch.from_numpy(val_features).float().to(device)
    val_labels_tensor = torch.from_numpy(val_labels).long().to(device)

    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    linear_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training linear classifier for {epochs} epochs...")
    best_acc = 0.0
    best_state_dict = classifier.state_dict()

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for features, labels in linear_train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            classifier.eval()
            with torch.no_grad():
                val_outputs = classifier(val_features_tensor)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == val_labels_tensor).float().mean().item()

                if accuracy >= best_acc:
                    best_acc = accuracy
                    best_state_dict = classifier.state_dict().copy()

                print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss/len(linear_train_loader):.4f}, "
                    f"Val Acc: {accuracy * 100:.2f}%"
                )

    print(f"\nLinear Probing Best Validation Accuracy: {best_acc * 100:.2f}%")
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


def _sort_key(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def get_eval_data_list(eval_cfg):
    data_cfg = getattr(eval_cfg, "data", None)
    assert data_cfg is not None, "evaluation.data must be defined."

    if hasattr(data_cfg, "keys"):
        keys = sorted(data_cfg.keys(), key=_sort_key)
        return [data_cfg[key] for key in keys]

    return list(data_cfg)


def build_dataset_loaders(eval_cfg, batch_size: int, num_workers: int):
    eval_transform = get_eval_transforms()
    dataset_loaders = {}
    dataset_sizes = {}

    for data_cfg in get_eval_data_list(eval_cfg):
        dataset_name = data_cfg.dataset_name
        dataset_loaders[dataset_name] = {}
        dataset_sizes[dataset_name] = {}

        splits = {
            "train": getattr(data_cfg, "train_split", "train"),
            "val": getattr(data_cfg, "val_split", "val"),
            "test": getattr(data_cfg, "test_split", "test"),
        }

        for split_label, split_name in splits.items():
            dataset = SubmissionDataset(
                root_dir=data_cfg.data_dir,
                split=split_name,
                transform=eval_transform,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=submission_collate_fn,
            )
            dataset_loaders[dataset_name][split_label] = loader
            dataset_sizes[dataset_name][split_label] = len(dataset)

    return dataset_loaders, dataset_sizes


def resolve_checkpoint_paths(eval_cfg):
    checkpoint_dir = Path(eval_cfg.checkpoint_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory not found: {checkpoint_dir}"

    checkpoint_entries = eval_cfg.get("checkpoints", [])
    assert checkpoint_entries, "evaluation.checkpoints must list at least one checkpoint."

    resolved_paths = []
    for entry in checkpoint_entries:
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = checkpoint_dir / entry
        assert entry_path.exists(), f"Checkpoint not found: {entry_path}"
        resolved_paths.append(entry_path)

    return checkpoint_dir, resolved_paths


def create_results_directory(results_dir: Path, experiment_id: str) -> Path:
    pattern = re.compile(rf"^{re.escape(experiment_id)}_(\d+)$")
    max_index = 0
    for item in results_dir.iterdir():
        match = pattern.match(item.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    target_dir = results_dir / f"{experiment_id}_{max_index + 1}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def write_summary(
    summary_path: Path,
    records,
    checkpoint_order,
    methods,
    dataset_names,
    dataset_sizes,
    linear_config=None,
    knn_config=None,
):
    checkpoint_map = defaultdict(lambda: defaultdict(dict))
    method_map = defaultdict(lambda: defaultdict(list))

    for rec in records:
        checkpoint_map[rec["checkpoint"]][rec["method"]][rec["dataset"]] = rec["accuracy"]
        method_map[rec["method"]][rec["dataset"]].append((rec["checkpoint"], rec["accuracy"]))

    lines = []
    lines.append("Submission Evaluation Summary")
    lines.append("=" * 80)

    # Add evaluation configuration
    lines.append("Evaluation Configuration:")
    if "linear" in methods and linear_config:
        lines.append(f"  Linear Probing:")
        lines.append(f"    - Optimizer: {linear_config.get('optimizer', 'SGD')}")
        lines.append(f"    - Learning Rate: {linear_config.get('lr', 0.001)}")
        lines.append(f"    - Momentum: {linear_config.get('momentum', 0.9)}")
        lines.append(f"    - Weight Decay: {linear_config.get('weight_decay', 0.0)}")
        lines.append(f"    - Batch Size: {linear_config.get('batch_size', 256)}")
        lines.append(f"    - Epochs: {linear_config.get('epochs', 100)}")
    if "knn" in methods and knn_config:
        lines.append(f"  k-NN:")
        lines.append(f"    - k: {knn_config.get('k', 20)}")
    lines.append("")

    lines.append("Datasets:")
    for dataset in dataset_names:
        sizes = dataset_sizes.get(dataset, {})
        size_bits = [f"{split}={sizes.get(split, 0)}" for split in SPLIT_ORDER]
        lines.append(f"- {dataset}: {', '.join(size_bits)}")

    lines.append("")
    lines.append("Hierarchy 1: Checkpoints > Methods > Datasets")
    for checkpoint_name in checkpoint_order:
        checkpoint_entry = checkpoint_map.get(checkpoint_name)
        if not checkpoint_entry:
            continue
        lines.append(f"- {checkpoint_name}")
        for method in methods:
            method_entry = checkpoint_entry.get(method)
            if not method_entry:
                continue
            lines.append(f"  - {METHOD_LABELS[method]}")
            for dataset in dataset_names:
                if dataset in method_entry:
                    lines.append(f"    - {dataset}: {method_entry[dataset] * 100:.2f}%")

    lines.append("")
    lines.append("Hierarchy 2: Methods > Datasets > Checkpoints")
    for method in methods:
        lines.append(f"- {METHOD_LABELS[method]}")
        dataset_entry = method_map.get(method, {})
        for dataset in dataset_names:
            checkpoint_scores = dataset_entry.get(dataset, [])
            if not checkpoint_scores:
                continue
            lines.append(f"  - {dataset}")
            for checkpoint_name, acc in sorted(checkpoint_scores, key=lambda x: x[1], reverse=True):
                lines.append(f"    - {checkpoint_name}: {acc * 100:.2f}%")

    lines.append("")
    lines.append("Detailed Records:")
    for rec in records:
        lines.append(
            f"- {rec['checkpoint']} | {METHOD_LABELS[rec['method']]} | {rec['dataset']}: "
            f"{rec['accuracy'] * 100:.2f}% -> {rec['prediction_path']}"
        )

    text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(text)

    return text


@hydra.main(version_base=None, config_path="../configs", config_name="submission")
def main(cfg: DictConfig):
    eval_cfg = cfg.evaluation
    assert eval_cfg is not None, "Evaluation config must be provided."

    print("=" * 80)
    print("Submission Evaluation")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    methods = [method.lower() for method in eval_cfg.get("methods", [])]
    assert methods, "Specify evaluation.methods in the config."
    for method in methods:
        assert method in SUPPORTED_METHODS, f"Unknown evaluation method '{method}'."
    print("Evaluation methods:", ", ".join(METHOD_LABELS[m] for m in methods))

    checkpoint_dir, checkpoint_paths = resolve_checkpoint_paths(eval_cfg)
    checkpoint_names = [path.name for path in checkpoint_paths]
    print(f"\nFound {len(checkpoint_paths)} checkpoints in: {checkpoint_dir}")
    for ckpt in checkpoint_names:
        print(f"  - {ckpt}")

    batch_size = eval_cfg.get("batch_size", cfg.training.batch_size)
    num_workers = cfg.training.num_workers
    dataset_loaders, dataset_sizes = build_dataset_loaders(eval_cfg, batch_size, num_workers)
    dataset_names = list(dataset_loaders.keys())
    assert dataset_names, "No evaluation datasets configured."

    print("\nDatasets:")
    for dataset in dataset_names:
        sizes = dataset_sizes[dataset]
        size_desc = ", ".join(f"{split}={sizes[split]}" for split in SPLIT_ORDER)
        print(f"  - {dataset}: {size_desc}")

    results_dir = Path(eval_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = extract_experiment_id(checkpoint_dir)
    experiment_results_dir = create_results_directory(results_dir, experiment_id)
    print(f"\nResults directory: {experiment_results_dir}")

    linear_epochs = eval_cfg.get("linear_probe_epochs", 100)
    linear_lr = eval_cfg.get("linear_probe_lr", 0.001)
    linear_batch_size = eval_cfg.get("linear_probe_batch_size", batch_size)
    knn_k = eval_cfg.get("knn_k", 20)

    # Store configuration for summary
    linear_config = {
        "optimizer": "SGD",
        "lr": linear_lr,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "batch_size": linear_batch_size,
        "epochs": linear_epochs,
    }
    knn_config = {
        "k": knn_k,
    }

    # Print evaluation configuration
    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    if "linear" in methods:
        print("Linear Probing Settings:")
        print(f"  - Optimizer: {linear_config['optimizer']}")
        print(f"  - Learning Rate: {linear_config['lr']}")
        print(f"  - Momentum: {linear_config['momentum']}")
        print(f"  - Weight Decay: {linear_config['weight_decay']}")
        print(f"  - Batch Size: {linear_config['batch_size']}")
        print(f"  - Epochs: {linear_config['epochs']}")
    if "knn" in methods:
        print("k-NN Settings:")
        print(f"  - k: {knn_config['k']}")

    records = []

    print("\n" + "=" * 80)
    print("BEGINNING EVALUATION")
    print("=" * 80)

    for method in methods:
        print("\n" + "=" * 80)
        print(f"METHOD: {METHOD_LABELS[method]}")
        print("=" * 80)

        if method == "knn":
            print(f"k-NN Configuration: k={knn_k}")
        else:
            print(f"Linear Probing Configuration:")
            print(f"  - Optimizer: {linear_config['optimizer']}")
            print(f"  - Learning Rate: {linear_config['lr']}")
            print(f"  - Momentum: {linear_config['momentum']}")
            print(f"  - Weight Decay: {linear_config['weight_decay']}")
            print(f"  - Batch Size: {linear_config['batch_size']}")
            print(f"  - Epochs: {linear_config['epochs']}")

        for checkpoint_path in checkpoint_paths:
            print("\n" + "-" * 80)
            print(f"Checkpoint: {checkpoint_path.name}")
            print("-" * 80)

            backbone = load_backbone(str(checkpoint_path), device)
            backbone.eval()

            for dataset_name in dataset_names:
                print(f"\nEvaluating on {dataset_name}...")

                # Extract features for this dataset
                dataset_features = {}
                for split_name in SPLIT_ORDER:
                    loader = dataset_loaders[dataset_name][split_name]
                    print(f"  Extracting {split_name} features...")
                    features, labels, filenames = extract_features(backbone, loader, device)
                    dataset_features[split_name] = {
                        "features": features,
                        "labels": labels,
                        "filenames": filenames,
                    }

                # Get features
                train_features = dataset_features["train"]["features"]
                train_labels = dataset_features["train"]["labels"]
                val_features = dataset_features["val"]["features"]
                val_labels = dataset_features["val"]["labels"]
                test_features = dataset_features["test"]["features"]
                test_filenames = dataset_features["test"]["filenames"]

                # Evaluate with current method
                if method == "knn":
                    knn_classifier = KNNClassifier(k=knn_k)
                    knn_classifier.train(train_features, train_labels)
                    val_acc = float(knn_classifier.evaluate(val_features, val_labels))
                    test_predictions = knn_classifier.predict(test_features)
                else:
                    num_classes = len(np.unique(train_labels))
                    linear_classifier, val_acc = linear_probe_training(
                        train_features,
                        train_labels,
                        val_features,
                        val_labels,
                        num_classes=num_classes,
                        device=device,
                        epochs=linear_epochs,
                        lr=linear_lr,
                        batch_size=linear_batch_size,
                    )
                    test_predictions = predict_with_classifier(linear_classifier, test_features, device)
                    del linear_classifier

                pred_filename = f"{checkpoint_path.stem}_{dataset_name}_{method}.csv"
                pred_path = experiment_results_dir / pred_filename
                save_predictions_to_csv(test_predictions, test_filenames, pred_path)

                method_label = METHOD_LABELS[method]
                print(
                    f"  [{checkpoint_path.name}] {method_label} - {dataset_name}: "
                    f"Validation Accuracy = {val_acc * 100:.2f}%"
                )
                print(f"  Predictions saved to: {pred_path}")

                records.append(
                    {
                        "checkpoint": checkpoint_path.name,
                        "method": method,
                        "dataset": dataset_name,
                        "accuracy": float(val_acc),
                        "prediction_path": str(pred_path),
                    }
                )

            del backbone
            torch.cuda.empty_cache()

    assert records, "No checkpoints were evaluated."

    summary_file = experiment_results_dir / "submission_results.txt"
    summary_text = write_summary(
        summary_file,
        records,
        checkpoint_names,
        methods,
        dataset_names,
        dataset_sizes,
        linear_config=linear_config if "linear" in methods else None,
        knn_config=knn_config if "knn" in methods else None,
    )

    print("\n" + summary_text)
    print(f"\n✓ Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
