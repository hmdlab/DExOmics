import os
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, silhouette_score
from sklearn.preprocessing import label_binarize
from network import *
from utils.data_tool import encode_label
from utils.model_utils import *

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluation")
parser.add_argument('tcga_cancer',      type=str)
parser.add_argument('encode_cell_line', type=str)
parser.add_argument('outloc',           type=str)
# epoch and l2_reg are now read automatically from the Optuna JSON;
# these optional overrides exist in case you want to run eval standalone.
parser.add_argument('-n', '--final_epochs', type=int, default=None,
                    help='Number of epochs saved during final retraining. '
                         'If omitted, read from best_optuna_params JSON.')
parser.add_argument('-reg', '--l2_reg', type=float, default=None,
                    help='Regularization parameter. '
                         'If omitted, read from best_optuna_params JSON.')
parser.add_argument('--umap-n-neighbors', type=int, default=15,
                    help='Number of neighbors used by UMAP (default: 15).')
parser.add_argument('--umap-min-dist', type=float, default=0.1,
                    help='Minimum distance used by UMAP (default: 0.1).')
args = parser.parse_args()

tcga_cancer      = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc           = args.outloc

if not os.path.exists(outloc):
    os.makedirs(outloc)

# ── Reproducibility ───────────────────────────────────────────────────────────
seed_value = 42
print("SEED VALUE:", seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Load Optuna best params ───────────────────────────────────────────────────
# Architecture params come from params.json (already updated by training script)
# Training params (epoch count, l2_reg) come from the Optuna output JSON
params_file   = f'../../data/modeling/{tcga_cancer}/params.json'
optuna_file   = f'{outloc}/best_optuna_params_{tcga_cancer}.json'

with open(params_file) as f:
    params = json.load(f)

with open(optuna_file) as f:
    optuna_results = json.load(f)

# Use CLI overrides if provided, otherwise fall back to Optuna JSON
epoch  = args.final_epochs if args.final_epochs is not None \
        else optuna_results["training"]["trial_epochs"]
l2_reg = args.l2_reg       if args.l2_reg       is not None \
        else optuna_results["training"]["l2_reg"]

print(f"Evaluating {epoch} epoch checkpoints  |  l2_reg={l2_reg}")
print(f"Architecture: {params}")

# ── Data loading ──────────────────────────────────────────────────────────────
data = pd.read_csv(f'../../results/TCGAprocessed/{tcga_cancer}/merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)
data = data.drop(columns=['DElabel'], axis=1)

mRNA_data_loc     = '../../results/rna_features/'
promoter_data_loc = '../../results/promoter_features/'

# ── Data split ────────────────────────────────────────────────────────────────
test_file  = f'../../data/modeling/{tcga_cancer}/full_test.csv'
val_file   = f'../../data/modeling/{tcga_cancer}/full_val.csv'
train_file = f'../../data/modeling/{tcga_cancer}/full_train.csv'

test  = pd.read_csv(test_file,  sep="\t", header=0).values[:, 0]
val   = pd.read_csv(val_file,   sep="\t", header=0).values[:, 0]
train = pd.read_csv(train_file, sep="\t", header=0).values[:, 0]

tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
tcga_test_df  = data.loc[data['Gene'].isin(test)].set_index('Gene')
tcga_val_df   = data.loc[data['Gene'].isin(val)].set_index('Gene')

merged_tcga_file = f'../../results/TCGAprocessed/{tcga_cancer}/merged_tcga_encode.csv'
(encode_Y_train, encode_Y_val, encode_Y_test,
encode_X_mRNA_train, encode_X_mRNA_val, encode_X_mRNA_test,
encode_X_promoter_train, encode_X_promoter_val,
encode_X_promoter_test) = prep_ml_data_split(
    merged_tcga_file=merged_tcga_file,
    mRNA_data_loc=mRNA_data_loc,
    promoter_data_loc=promoter_data_loc,
    cell_line=encode_cell_line,
    train_file=train_file,
    val_file=val_file,
    test_file=test_file,
    outloc=outloc)

# ── Standardization ───────────────────────────────────────────────────────────
scaler = StandardScaler()
tcga_train_df.iloc[:, :-1] = scaler.fit_transform(tcga_train_df.iloc[:, :-1])
tcga_test_df.iloc[:,  :-1] = scaler.transform(tcga_test_df.iloc[:, :-1])
tcga_val_df.iloc[:,   :-1] = scaler.transform(tcga_val_df.iloc[:, :-1])

tcga_train_data, tcga_train_labels = tcga_train_df.iloc[:, :-1], tcga_train_df.iloc[:, -1]
tcga_val_data,   tcga_val_labels   = tcga_val_df.iloc[:,   :-1], tcga_val_df.iloc[:,   -1]
tcga_test_data,  tcga_test_labels  = tcga_test_df.iloc[:,  :-1], tcga_test_df.iloc[:,  -1]

# ── Batch initialization ──────────────────────────────────────────────────────
train_steps, train_batches = batch_iter(
    tcga_train_data,
    encode_X_mRNA_train.values[:, 1],
    encode_X_promoter_train.values[:, 1],
    encode_Y_train.values,
    batch_size=len(tcga_train_data), shuffle=False)

val_steps, val_batches = batch_iter(
    tcga_val_data,
    encode_X_mRNA_val.values[:, 1],
    encode_X_promoter_val.values[:, 1],
    encode_Y_val.values,
    batch_size=len(tcga_val_data), shuffle=False)

test_steps, test_batches = batch_iter(
    tcga_test_data,
    encode_X_mRNA_test.values[:, 1],
    encode_X_promoter_test.values[:, 1],
    encode_Y_test.values,
    batch_size=len(tcga_test_data), shuffle=False)

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss()
device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Evaluate best_model.pth on val set ───────────────────────────────────────
best_model = ConcatedNet(params).to(device)
best_model.load_state_dict(torch.load(f'{outloc}/best_model.pth', map_location=device))
val_auc_roc, val_accuracy, val_f1, val_pr_auc, val_recalls, val_cm, val_loss = evaluate(
    best_model, val_steps, val_batches, criterion, device)
print(f'Best model on Val: AUC-ROC={val_auc_roc:.4f}, ACC={val_accuracy:.4f}, '
    f'F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f}, RECALLS={val_recalls}, Loss={val_loss:.4f}')
print("Validation Confusion Matrix:")
print(val_cm)

# ── Evaluate all epoch checkpoints on test set ────────────────────────────────
metrics_dir = '../../results/metrics'
os.makedirs(metrics_dir, exist_ok=True)

test_AUCs = []
test_ACCs = []

with open(f'{metrics_dir}/{tcga_cancer}_full_test_metrics.csv', 'w') as file:
    print(f'Testing  |  l2_reg={l2_reg}  |  final_epochs={epoch}', file=file)

    for i in range(epoch):
        ckpt_path = f'{outloc}epoch{i+1}.pth'
        if not os.path.exists(ckpt_path):
            print(f'  [WARNING] checkpoint not found, skipping: {ckpt_path}')
            continue

        model = ConcatedNet(params).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_auc_roc, test_accuracy, test_f1, test_pr_auc, test_recalls, test_cm, test_loss = evaluate(
            model, test_steps, test_batches, criterion, device)
        test_AUCs.append(test_auc_roc)
        test_ACCs.append(test_accuracy)
        print(f'Test {i+1}/{epoch}: AUC-ROC={test_auc_roc:.4f}, ACC={test_accuracy:.4f}, '
            f'F1={test_f1:.4f}, PR-AUC={test_pr_auc:.4f}, RECALLS={test_recalls}, Loss={test_loss:.4f}',
            file=file)

    # Best model on test set
    test_auc_roc, test_accuracy, test_f1, test_pr_auc, test_recalls, test_cm, test_loss = evaluate(
        best_model, test_steps, test_batches, criterion, device)
    print(f'Best model on Test: AUC-ROC={test_auc_roc:.4f}, ACC={test_accuracy:.4f}, '
        f'F1={test_f1:.4f}, PR-AUC={test_pr_auc:.4f}, RECALLS={test_recalls}, Loss={test_loss:.4f}', file=file)

print(f'Best model on Test: AUC-ROC={test_auc_roc:.4f}, ACC={test_accuracy:.4f}, '
    f'F1={test_f1:.4f}, PR-AUC={test_pr_auc:.4f}, RECALLS={test_recalls}')
print('confusion matrix: ', test_cm)

# ── Visualization ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 18, 'axes.titlesize': 12,
    'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
})

plot_dir = f'../../plots_{tcga_cancer}/concat_model'
os.makedirs(plot_dir, exist_ok=True)

epochs_x = np.arange(1, len(test_AUCs) + 1).tolist()

plt.figure()
plt.plot(epochs_x, test_AUCs)
plt.title('AUC over epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{plot_dir}/test_AUCs.png', dpi=300)
plt.close()

plt.figure()
plt.plot(epochs_x, test_ACCs)
plt.title('ACC over epochs')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{plot_dir}/test_ACCs.png', dpi=300)
plt.close()

# ── ROC curve (OvR) ───────────────────────────────────────────────────────────
train_y_true, train_y_prob, train_y_pred = model_prob(best_model, train_steps, train_batches, device)
val_y_true,   val_y_prob,   val_y_pred   = model_prob(best_model, val_steps,   val_batches,   device)
test_y_true,  test_y_prob,  test_y_pred  = model_prob(best_model, test_steps,  test_batches,  device)

test_y_true_bin = label_binarize(test_y_true, classes=[0, 1, 2])
n_classes = test_y_prob.shape[1]

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y_true_bin[:, i], test_y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 8))
colors = ['lightblue', 'lightgreen', 'lightsalmon']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i],
            label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'{plot_dir}/test_roc_curve_ovr.png', dpi=300)
plt.close()

# ── Confusion matrices ────────────────────────────────────────────────────────
class_names = ['nonDEG', 'downDEG', 'upDEG']
tick_pos    = [0.5, 1.5, 2.5]

for split, y_true, y_pred in [('train', train_y_true, train_y_pred),
                                ('val',   val_y_true,   val_y_pred),
                                ('test',  test_y_true,  test_y_pred)]:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{split.capitalize()} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(tick_pos, class_names)
    plt.yticks(tick_pos, class_names, va='center')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{split}_cm.png', dpi=300)
    plt.close()

# ── UMAP of the final embedding ───────────────────────────────────────────────
numba_cache_dir = os.path.join(outloc, '.numba_cache')
os.makedirs(numba_cache_dir, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', numba_cache_dir)

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "UMAP plotting requires umap-learn. Install it with: pip install umap-learn"
    ) from exc


def plot_embedding_umap(model, data_steps, data_batches, split, output_dir):
    embeddings, labels = extract_embeddings(
        model, data_steps, data_batches, device
    )
    if embeddings.ndim != 2 or embeddings.shape[0] < 3:
        raise ValueError(
            f'{split}: expected at least three 2D embedding rows, got '
            f'{embeddings.shape}'
        )

    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    n_neighbors = min(args.umap_n_neighbors, embeddings.shape[0] - 1)
    if n_neighbors < 2:
        raise ValueError(f'{split}: UMAP requires at least three samples')

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric='euclidean',
        random_state=seed_value,
    )
    coordinates = reducer.fit_transform(scaled_embeddings)

    present_classes = np.unique(labels)
    embedding_silhouette = np.nan
    if 1 < len(present_classes) < len(labels):
        embedding_silhouette = silhouette_score(scaled_embeddings, labels)

    umap_df = pd.DataFrame({
        'UMAP1': coordinates[:, 0],
        'UMAP2': coordinates[:, 1],
        'label': labels.astype(int),
    })
    label_to_name = dict(enumerate(class_names))
    umap_df['class'] = umap_df['label'].map(label_to_name)
    umap_df.to_csv(f'{output_dir}/{split}_embedding_umap.csv', index=False)

    palette = {
        'nonDEG': '#4C78A8',
        'downDEG': '#E45756',
        'upDEG': '#59A14F',
    }
    plt.figure(figsize=(8, 7))
    sns.scatterplot(
        data=umap_df,
        x='UMAP1',
        y='UMAP2',
        hue='class',
        hue_order=class_names,
        palette=palette,
        alpha=0.7,
        s=35,
        linewidth=0,
    )
    score_text = (
        f'{embedding_silhouette:.3f}'
        if np.isfinite(embedding_silhouette)
        else 'NA'
    )
    plt.title(
        f'{split.capitalize()} final-layer embedding UMAP\n'
        f'Embedding silhouette = {score_text}'
    )
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='DE class', frameon=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{split}_embedding_umap.png', dpi=300)
    plt.close()

    return {
        'split': split,
        'n_samples': len(labels),
        'embedding_dim': embeddings.shape[1],
        'silhouette_score': embedding_silhouette,
    }


umap_summary = []
for split, steps, batches in [
    ('train', train_steps, train_batches),
    ('val', val_steps, val_batches),
    ('test', test_steps, test_batches),
]:
    umap_summary.append(
        plot_embedding_umap(best_model, steps, batches, split, plot_dir)
    )

pd.DataFrame(umap_summary).to_csv(
    f'{plot_dir}/embedding_umap_summary.csv', index=False
)
print(f'UMAP plots and coordinates saved to: {plot_dir}')
