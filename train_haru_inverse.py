"""Training pipeline for the FacialExpressionGCN classifier.

Loads the merged landmark dataset (npz with 226-feature vectors and
routine-id labels), performs a stratified train/val/test split that
gracefully handles small classes, trains the GCN, saves the best
checkpoint, and writes a battery of plots and reports under ``plots/``.
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from expression_model import FacialExpressionGCN

# =============================================================================
# Configuration
# =============================================================================

ROUTINE_IDS = [
    2092, 2083, 2068, 2057, 2006, 2037,
    2036, 2087, 2010, 2071, 2073, 2023,
    2081, 2021, 2067, 2059, 2017,
]
NUM_CLASSES = len(ROUTINE_IDS)

DATA_PATH = 'data/merged_training_data.npz'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

_MIN_SAMPLES_FOR_STRATIFIED_SPLIT = 3
_INPUT_FEATURE_DIM = 226
_EARLY_STOPPING_PATIENCE = 50

# Augmentation parameters.
_AUG_TRANSLATION_STD = 0.02
_AUG_SCALE_RANGE = (0.95, 1.05)
_AUG_ROTATION_DEG = 5.0
_AUG_NOISE_STD = 0.01


# =============================================================================
# Dataset
# =============================================================================

class HaruInverseDataset(Dataset):
    """Wraps the npz dataset and (optionally) applies augmentation."""

    def __init__(self, npz_file: str, indices: np.ndarray | None = None,
                 train: bool = True) -> None:
        data = np.load(npz_file, allow_pickle=True)
        all_landmarks = data['landmarks']
        all_routine_ids = np.array([int(r) for r in data['routines']])

        if indices is not None:
            self.landmarks = all_landmarks[indices]
            self.routine_ids = all_routine_ids[indices]
        else:
            self.landmarks = all_landmarks
            self.routine_ids = all_routine_ids

        self.id_to_idx = {rid: idx for idx, rid in enumerate(ROUTINE_IDS)}
        self.idx_to_id = {idx: rid for rid, idx in self.id_to_idx.items()}
        self.mapped_ids = np.array(
            [self.id_to_idx.get(int(r), 0) for r in self.routine_ids])

        self.train = train

        print(f'Dataset loaded: {len(self.landmarks)} samples, '
              f'{len(self.id_to_idx)} classes.')
        if train:
            self._print_class_distribution()
            self._calculate_class_weights()

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        y = torch.tensor(self.mapped_ids[idx], dtype=torch.long)
        if self.train:
            x = self._apply_augmentation(x)
        return x, y

    def get_labels(self) -> np.ndarray:
        return self.mapped_ids

    def get_class_weights(self) -> torch.Tensor | None:
        return getattr(self, 'class_weights', None)

    # ---------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------

    def _print_class_distribution(self) -> None:
        unique_classes, class_counts = np.unique(self.mapped_ids,
                                                 return_counts=True)
        print('\nClass distribution:')
        for cls, count in zip(unique_classes, class_counts):
            print(f'  Routine {self.idx_to_id[cls]} (class {cls}): '
                  f'{count} samples')

    def _calculate_class_weights(self) -> None:
        _, class_counts = np.unique(self.mapped_ids, return_counts=True)
        max_count = int(np.max(class_counts))
        self.class_weights = torch.FloatTensor(
            [max_count / c for c in class_counts])

    @staticmethod
    def _apply_augmentation(x: torch.Tensor) -> torch.Tensor:
        """Random shift, scale, rotation, and Gaussian noise on (113, 2) points."""
        x = x.view(-1, 2)

        # Shift.
        x = x + torch.randn(2) * _AUG_TRANSLATION_STD

        # Scale.
        scale = torch.rand(1) * (_AUG_SCALE_RANGE[1] - _AUG_SCALE_RANGE[0]) \
            + _AUG_SCALE_RANGE[0]
        x = x * scale

        # Rotation.
        angle = torch.rand(1) * (2 * _AUG_ROTATION_DEG) - _AUG_ROTATION_DEG
        rad = torch.deg2rad(angle)
        rot = torch.tensor([[torch.cos(rad), -torch.sin(rad)],
                            [torch.sin(rad), torch.cos(rad)]])
        x = x @ rot.T

        # Noise.
        x = x + torch.randn_like(x) * _AUG_NOISE_STD
        return x.view(-1)


# =============================================================================
# Stratified split
# =============================================================================

def stratified_split_dataset(npz_file: str,
                             val_split: float = 0.2,
                             test_split: float = 0.2,
                             random_state: int = 42):
    """Splits the dataset stratified by class, with manual handling of tiny classes."""
    base = HaruInverseDataset(npz_file, train=False)
    labels = base.get_labels()
    indices = np.arange(len(base))

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f'\nStratified split (min samples per class = '
          f'{_MIN_SAMPLES_FOR_STRATIFIED_SPLIT}):')
    for label, count in zip(unique_labels, counts):
        ok = '✓' if count >= _MIN_SAMPLES_FOR_STRATIFIED_SPLIT else '✗'
        print(f'  Class {label} (Routine {base.idx_to_id[label]}): '
              f'{count} samples {ok}')

    small_idx, small_lbl, large_idx, large_lbl = _separate_by_class_size(
        labels, indices, unique_labels, counts)

    train_lg, val_lg, test_lg = _split_large_classes(
        large_idx, large_lbl, val_split, test_split, random_state)
    train_sm, val_sm, test_sm = _split_small_classes(
        small_idx, small_lbl, random_state)

    train_indices = train_lg + train_sm
    val_indices = val_lg + val_sm
    test_indices = test_lg + test_sm

    train_ds = HaruInverseDataset(npz_file, indices=train_indices, train=True)
    val_ds = HaruInverseDataset(npz_file, indices=val_indices, train=False)
    test_ds = HaruInverseDataset(npz_file, indices=test_indices, train=False)

    print(f'\nSplit sizes: train={len(train_ds)}, '
          f'val={len(val_ds)}, test={len(test_ds)}')
    for name, ds in (('train', train_ds), ('val', val_ds), ('test', test_ds)):
        print(f'\n{name}:')
        counter = Counter(ds.get_labels())
        for label in sorted(counter):
            print(f'  class {label} (routine {ds.idx_to_id[label]}): '
                  f'{counter[label]}')

    return train_ds, val_ds, test_ds


def _separate_by_class_size(labels, indices, unique_labels, counts):
    small_idx, small_lbl = [], []
    large_idx, large_lbl = [], []
    for label, count in zip(unique_labels, counts):
        mask = labels == label
        cls_indices = indices[mask]
        if count < _MIN_SAMPLES_FOR_STRATIFIED_SPLIT:
            small_idx.extend(cls_indices)
            small_lbl.extend([label] * count)
        else:
            large_idx.extend(cls_indices)
            large_lbl.extend([label] * count)
    return small_idx, small_lbl, large_idx, large_lbl


def _split_large_classes(indices, labels, val_split, test_split, random_state):
    if not indices:
        return [], [], []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split,
                                 random_state=random_state)
    trv_idx, te_idx = next(sss.split(indices, labels))

    trv_indices = [indices[i] for i in trv_idx]
    trv_labels = [labels[i] for i in trv_idx]
    adjusted_val = val_split / (1 - test_split)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val,
                                  random_state=random_state)
    tr_idx, val_idx = next(sss2.split(trv_indices, trv_labels))

    return (
        [trv_indices[i] for i in tr_idx],
        [trv_indices[i] for i in val_idx],
        [indices[i] for i in te_idx],
    )


def _split_small_classes(indices, labels, random_state):
    np.random.seed(random_state)
    train, val, test = [], [], []
    for label in np.unique(labels):
        mask = np.array(labels) == label
        cls_indices = np.array(indices)[mask]
        np.random.shuffle(cls_indices)
        n = len(cls_indices)
        if n == 1:
            train.extend(cls_indices)
        elif n == 2:
            train.append(cls_indices[0])
            val.append(cls_indices[1])
        else:
            test.append(cls_indices[0])
            val.append(cls_indices[1])
            train.extend(cls_indices[2:])
    return train, val, test


# =============================================================================
# Evaluation helpers
# =============================================================================

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == targets).sum().item() / targets.size(0)


def evaluate_model(model: nn.Module, loader: DataLoader,
                   device: torch.device, loss_fn: nn.Module):
    """Returns (avg_loss, avg_accuracy, all_preds, all_targets)."""
    model.eval()
    total_loss = total_acc = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            total_loss += loss_fn(outputs, y).item() * x.size(0)
            total_acc += calculate_accuracy(outputs, y) * x.size(0)
            total_samples += x.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return (total_loss / total_samples,
            total_acc / total_samples,
            all_preds, all_targets)


# =============================================================================
# Plots
# =============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          title: str = 'Confusion Matrix',
                          filename: str = 'confusion_matrix.png') -> None:
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)

    threshold = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]
            text = '0' if value == 0 else f'{value:.2f}'
            plt.text(j, i, text, ha='center',
                     color='white' if value > threshold else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_curve(filename: str, ylabel: str, title: str,
                series: list[tuple[list[float], str, str]],
                best_epoch: int) -> None:
    """series: list of (values, label, color)."""
    plt.figure(figsize=(12, 8))
    for values, label, color in series:
        plt.plot(values, label=label, color=color, linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch + 1})')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_history(history: dict, best_epoch: int, plots_dir: str) -> None:
    """Writes all per-curve plots driven by the training history dict."""
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    test_loss = history['test_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    test_acc = history['test_acc']

    plots = [
        ('train_val_loss_curves.png', 'Loss', 'Training and Validation Loss',
         [(train_loss, 'Training Loss', 'blue'),
          (val_loss, 'Validation Loss', 'orange')]),
        ('test_loss_curve.png', 'Loss', 'Test Loss',
         [(test_loss, 'Test Loss', 'red')]),
        ('train_val_accuracy_curves.png', 'Accuracy',
         'Training and Validation Accuracy',
         [(train_acc, 'Training Accuracy', 'blue'),
          (val_acc, 'Validation Accuracy', 'orange')]),
        ('test_accuracy_curve.png', 'Accuracy', 'Test Accuracy',
         [(test_acc, 'Test Accuracy', 'red')]),
        ('train_val_error_curves.png', 'Error Rate',
         'Training and Validation Error Rate',
         [([1 - a for a in train_acc], 'Training Error', 'blue'),
          ([1 - a for a in val_acc], 'Validation Error', 'orange')]),
        ('test_error_curve.png', 'Error Rate', 'Test Error Rate',
         [([1 - a for a in test_acc], 'Test Error', 'red')]),
    ]
    for filename, ylabel, title, series in plots:
        _plot_curve(os.path.join(plots_dir, filename),
                    ylabel, title, series, best_epoch)

    # Combined 3-up summary.
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    for vals, lbl, color in [(train_loss, 'Training Loss', 'blue'),
                             (val_loss, 'Validation Loss', 'orange'),
                             (test_loss, 'Test Loss', 'red')]:
        plt.plot(vals, label=lbl, color=color, linewidth=2,
                 alpha=0.7 if lbl == 'Test Loss' else 1.0)
    plt.axvline(x=best_epoch, color='green', linestyle='--',
                label=f'Best Epoch ({best_epoch + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    for vals, lbl, color in [(train_acc, 'Training Accuracy', 'blue'),
                             (val_acc, 'Validation Accuracy', 'orange'),
                             (test_acc, 'Test Accuracy', 'red')]:
        plt.plot(vals, label=lbl, color=color, linewidth=2,
                 alpha=0.7 if lbl == 'Test Accuracy' else 1.0)
    plt.axvline(x=best_epoch, color='green', linestyle='--',
                label=f'Best Epoch ({best_epoch + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    metrics = ['Loss', 'Accuracy']
    bars = [
        ('Training', 'blue', [train_loss[best_epoch], train_acc[best_epoch]]),
        ('Validation', 'orange', [val_loss[best_epoch], val_acc[best_epoch]]),
        ('Test', 'red', [test_loss[best_epoch], test_acc[best_epoch]]),
    ]
    x_pos = np.arange(len(metrics))
    width = 0.25
    for offset, (label, color, vals) in zip((-1, 0, 1), bars):
        plt.bar(x_pos + offset * width, vals, width,
                label=label, alpha=0.8, color=color)
    plt.xticks(x_pos, metrics)
    plt.ylabel('Value')
    plt.title(f'Best epoch performance ({best_epoch + 1})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Training loop
# =============================================================================

def train_model_improved(epochs: int = 1000,
                         batch_size: int = 64,
                         lr: float = 1e-4,
                         dropout_rate: float = 0.5,
                         weight_decay: float = 1e-3,
                         val_split: float = 0.15,
                         test_split: float = 0.15) -> nn.Module:
    """Trains the GCN classifier and writes plots/reports under PLOTS_DIR."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    train_ds, val_ds, test_ds = stratified_split_dataset(
        DATA_PATH, val_split=val_split, test_split=test_split, random_state=42)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    model = FacialExpressionGCN(input_size=_INPUT_FEATURE_DIM,
                                num_classes=NUM_CLASSES,
                                dropout_rate=dropout_rate).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7)

    class_weights = train_ds.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print('Using class-weighted loss based on training distribution.')
    else:
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': [],
        'best_epoch': 0,
    }
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0

    print(f'\nTraining (early stopping patience={_EARLY_STOPPING_PATIENCE}):')
    label_names = [str(train_ds.idx_to_id[i]) for i in range(NUM_CLASSES)]

    for epoch in range(epochs):
        train_loss, train_acc = _run_train_epoch(
            model, train_loader, device, optimizer, ce_loss)
        val_loss, val_acc, val_preds, val_targets = evaluate_model(
            model, val_loader, device, ce_loss)
        test_loss, test_acc, test_preds, test_targets = evaluate_model(
            model, test_loader, device, ce_loss)

        for key, value in (('train_loss', train_loss), ('train_acc', train_acc),
                           ('val_loss', val_loss), ('val_acc', val_acc),
                           ('test_loss', test_loss), ('test_acc', test_acc)):
            history[key].append(value)

        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0 or epoch < 10:
            print(f'  Epoch [{epoch + 1}/{epochs}]: '
                  f'train_loss={train_loss:.6f} train_acc={train_acc:.4f} '
                  f'val_loss={val_loss:.6f} val_acc={val_acc:.4f} '
                  f'test_loss={test_loss:.6f} test_acc={test_acc:.4f} '
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, 'best_val_loss_model.pth'))
            plot_confusion_matrix(
                confusion_matrix(val_targets, val_preds), label_names,
                'Best Validation Confusion Matrix',
                os.path.join(PLOTS_DIR, 'best_val_confusion_matrix.png'))
            plot_confusion_matrix(
                confusion_matrix(test_targets, test_preds), label_names,
                f'Test Confusion Matrix (Epoch {epoch + 1})',
                os.path.join(PLOTS_DIR, 'best_val_test_confusion_matrix.png'))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, 'best_val_acc_model.pth'))
        else:
            patience += 1

        if patience >= _EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping at epoch {epoch + 1}.')
            break

    history['best_epoch'] = best_epoch
    print(f'\nDone. Best val loss={best_val_loss:.6f} acc={best_val_acc:.4f} '
          f'at epoch {best_epoch + 1}.')

    _plot_history(history, best_epoch, PLOTS_DIR)
    best_model = _evaluate_best_model(test_loader, device, ce_loss,
                                      label_names, best_epoch, history,
                                      train_ds, val_ds, test_ds, dropout_rate)
    return best_model


def _run_train_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss = total_acc = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc += calculate_accuracy(outputs, y) * x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples, total_acc / total_samples


def _evaluate_best_model(test_loader, device, loss_fn, label_names,
                         best_epoch, history, train_ds, val_ds, test_ds,
                         dropout_rate) -> nn.Module:
    print('\nEvaluating best checkpoint on the test set...')
    best = FacialExpressionGCN(input_size=_INPUT_FEATURE_DIM,
                               num_classes=NUM_CLASSES,
                               dropout_rate=dropout_rate).to(device)
    best.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'best_val_loss_model.pth')))

    test_loss, test_acc, preds, targets = evaluate_model(
        best, test_loader, device, loss_fn)
    print(f'Best model: test_loss={test_loss:.6f} test_acc={test_acc:.4f}')

    plot_confusion_matrix(
        confusion_matrix(targets, preds), label_names,
        'Final Test Confusion Matrix',
        os.path.join(PLOTS_DIR, 'final_test_confusion_matrix.png'))

    test_report = classification_report(
        targets, preds, target_names=label_names, zero_division=0)
    print(f'\nTest report:\n{test_report}')

    full_ds = HaruInverseDataset(DATA_PATH, train=False)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)
    full_loss, full_acc, full_preds, full_targets = evaluate_model(
        best, full_loader, device, loss_fn)
    print(f'Full-data: loss={full_loss:.6f} acc={full_acc:.4f}')

    plot_confusion_matrix(
        confusion_matrix(full_targets, full_preds), label_names,
        'Final Confusion Matrix (All Data)',
        os.path.join(PLOTS_DIR, 'final_confusion_matrix.png'))

    np.save(os.path.join(PLOTS_DIR, 'training_history.npy'), history)

    _write_test_report(test_report, test_loss, test_acc, history, best_epoch)
    _write_history_text(history, best_epoch, best_val_loss=history['val_loss'][best_epoch],
                        best_val_acc=history['val_acc'][best_epoch],
                        test_loss=test_loss, test_acc=test_acc)
    _write_summary_report(train_ds, val_ds, test_ds, history,
                          best_epoch, test_loss, test_acc,
                          full_loss, full_acc, test_report)

    torch.save(best.state_dict(),
               os.path.join(MODELS_DIR, 'expression_mapping.pth'))
    return best


def _write_test_report(report: str, test_loss: float, test_acc: float,
                       history: dict, best_epoch: int) -> None:
    path = os.path.join(PLOTS_DIR, 'test_classification_report.txt')
    with open(path, 'w') as f:
        f.write('Best model — test set performance\n')
        f.write('=' * 35 + '\n\n')
        f.write(f'Best epoch: {best_epoch + 1}\n')
        f.write(f'Test loss:     {test_loss:.6f}\n')
        f.write(f'Test accuracy: {test_acc:.4f}\n\n')
        f.write('Per-class report:\n')
        f.write(report)
        f.write('\nTrain/val/test comparison:\n')
        f.write(f'  Train: loss={history["train_loss"][best_epoch]:.6f}  '
                f'acc={history["train_acc"][best_epoch]:.4f}\n')
        f.write(f'  Val:   loss={history["val_loss"][best_epoch]:.6f}  '
                f'acc={history["val_acc"][best_epoch]:.4f}\n')
        f.write(f'  Test:  loss={test_loss:.6f}  acc={test_acc:.4f}\n')


def _write_history_text(history: dict, best_epoch: int,
                        best_val_loss: float, best_val_acc: float,
                        test_loss: float, test_acc: float) -> None:
    path = os.path.join(PLOTS_DIR, 'training_history.txt')
    with open(path, 'w') as f:
        f.write('Training history\n')
        f.write('=' * 30 + '\n\n')
        f.write(f'Best epoch: {best_epoch + 1}\n')
        f.write(f'Best val loss:     {best_val_loss:.6f}\n')
        f.write(f'Best val accuracy: {best_val_acc:.4f}\n\n')
        f.write(f'Final test loss:     {test_loss:.6f}\n')
        f.write(f'Final test accuracy: {test_acc:.4f}\n\n')

        header = f'{"Epoch":<8}{"TrLoss":<14}{"TrAcc":<14}'\
                 f'{"VaLoss":<14}{"VaAcc":<14}{"TeLoss":<14}{"TeAcc":<14}\n'
        f.write(header)
        f.write('-' * 92 + '\n')
        for i in range(len(history['train_loss'])):
            f.write(f'{i + 1:<8}{history["train_loss"][i]:<14.6f}'
                    f'{history["train_acc"][i]:<14.4f}'
                    f'{history["val_loss"][i]:<14.6f}'
                    f'{history["val_acc"][i]:<14.4f}'
                    f'{history["test_loss"][i]:<14.6f}'
                    f'{history["test_acc"][i]:<14.4f}\n')


def _write_summary_report(train_ds, val_ds, test_ds, history,
                          best_epoch, test_loss, test_acc,
                          full_loss, full_acc, test_report) -> None:
    path = os.path.join(PLOTS_DIR, 'training_report.txt')
    with open(path, 'w') as f:
        f.write('Training summary\n')
        f.write('=' * 25 + '\n\n')
        f.write(f'Train: {len(train_ds)} samples\n')
        f.write(f'Val:   {len(val_ds)} samples\n')
        f.write(f'Test:  {len(test_ds)} samples\n\n')
        f.write(f'Best model (epoch {best_epoch + 1}):\n')
        f.write(f'  Train: loss={history["train_loss"][best_epoch]:.6f}  '
                f'acc={history["train_acc"][best_epoch]:.4f}\n')
        f.write(f'  Val:   loss={history["val_loss"][best_epoch]:.6f}  '
                f'acc={history["val_acc"][best_epoch]:.4f}\n')
        f.write(f'  Test:  loss={test_loss:.6f}  acc={test_acc:.4f}\n')
        f.write(f'  Full:  loss={full_loss:.6f}  acc={full_acc:.4f}\n\n')
        f.write('Test classification report:\n')
        f.write(test_report)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    train_model_improved(epochs=1000, batch_size=64, lr=1e-4,
                         dropout_rate=0.5, weight_decay=1e-3,
                         val_split=0.15, test_split=0.15)
