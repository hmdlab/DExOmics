import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from network import *
from utils.data_tool import *
from utils.model_utils import *
from collections import Counter

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Training with Optuna auto-tuning")
parser.add_argument('tcga_cancer',      type=str)
parser.add_argument('encode_cell_line', type=str)
parser.add_argument('outloc',           type=str)
parser.add_argument('--n_trials',     type=int, default=30,
                    help='Number of Optuna trials (default: 30)')
parser.add_argument('--final_epochs', type=int, default=100,
                    help='Epochs for final retraining with best params (default: 100)')
args = parser.parse_args()

tcga_cancer      = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc           = args.outloc
N_TRIALS         = args.n_trials
FINAL_EPOCHS     = args.final_epochs

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data loading ──────────────────────────────────────────────────────────────
data = pd.read_csv(f'../../results/TCGAprocessed/{tcga_cancer}/merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)
data = data.drop(columns=['DElabel'], axis=1)

mRNA_data_loc     = '../../results/rna_features/'
promoter_data_loc = '../../results/promoter_features/'
params_file       = f'../../data/modeling/{tcga_cancer}/params.json'

# ── Data split ────────────────────────────────────────────────────────────────
train_names = f'../../data/modeling/{tcga_cancer}/full_train.csv'
val_names   = f'../../data/modeling/{tcga_cancer}/full_val.csv'
test_names  = f'../../data/modeling/{tcga_cancer}/full_test.csv'

train = pd.read_csv(train_names, sep="\t", header=0).values[:, 0]
val   = pd.read_csv(val_names,   sep="\t", header=0).values[:, 0]

tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
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
    train_file=train_names,
    val_file=val_names,
    test_file=test_names,
    outloc=outloc)

# ── TCGA Standardization ──────────────────────────────────────────────────────
scaler = StandardScaler()
tcga_train_df.iloc[:, :-1] = scaler.fit_transform(tcga_train_df.iloc[:, :-1])
tcga_val_df.iloc[:,   :-1] = scaler.transform(tcga_val_df.iloc[:, :-1])

tcga_train_data,  tcga_train_labels = tcga_train_df.iloc[:, :-1], tcga_train_df.iloc[:, -1]
tcga_val_data,    tcga_val_labels   = tcga_val_df.iloc[:, :-1], tcga_val_df.iloc[:, -1]

# ── Fixed params (determined by data shape, never tuned) ─────────────────────
FIXED_PARAMS = {
    "mRNA_in_channels":     1,
    "promoter_in_channels": 1,
    "n_feature_mRNA":       int(encode_X_mRNA_train.values[:, 1][0].shape[0]),
    "n_feature_promoter":   int(encode_X_promoter_train.values[:, 1][0].shape[0]),
    "tcga_input_size":      int(tcga_train_df.shape[1] - 1),
    "n_out":                3,
}
print(encode_X_mRNA_train.values[:, 1][0].shape)
print("dim0:", encode_X_mRNA_train.values[:, 1][0].shape[0])
print("dim1:", encode_X_mRNA_train.values[:, 1][0].shape[1])

# ── Class weights (constant across all trials) ────────────────────────────────
weights = compute_class_weight(class_weight="balanced",
                                    classes=np.unique(tcga_train_labels),
                                    y=tcga_train_labels)
# Amplify the weight of downDEGs
# weights = [0.35, 160, 6.77]                                   
class_weights = torch.FloatTensor(weights)
print(class_weights)

print(f"\nFixed params: {FIXED_PARAMS}")
print("Data loading done.\n")


min_seq_len_rna = min(mat.shape[1] for mat in encode_X_mRNA_train.values[:, 1])
min_seq_len_dna = min(mat.shape[1] for mat in encode_X_promoter_train.values[:, 1])
print(f"Min RNA seq len: {min_seq_len_rna}, Min DNA seq len: {min_seq_len_dna}")

def build_params(trial_or_dict, is_trial=True):
    """
    Construct the full params dict for ConcatedNet.

    Dimension constraints derived from network.py forward():
    - After RNA_layers conv1: output channels = RNA_n_channel_1st
    - After DNA_layers conv1: output channels = DNA_n_channel_1st
    - x_encode_combined = cat(x_rna, x_dna) → size = RNA_n_channel_1st + DNA_n_channel_1st
    - fc_layers in_features MUST equal encode_last_n_channel
        → so encode_last_n_channel is DERIVED, not independently sampled
    - final cat(x_tcga, x_encode_combined) → size = tcga_hidden_size + encode_last_n_channel
    - output_layer in_features MUST equal last_n_channel
        → so last_n_channel is also DERIVED
    """
    if is_trial:
        t = trial_or_dict
        RNA_n_channel_1st = t.suggest_categorical("RNA_n_channel_1st", [16, 32, 64])
        DNA_n_channel_1st = t.suggest_categorical("DNA_n_channel_1st", [16, 32, 64])
        RNA_n_ConvLayer   = t.suggest_int("RNA_n_ConvLayer", 1, 3)
        DNA_n_ConvLayer   = t.suggest_int("DNA_n_ConvLayer", 1, 3)
        # Cap kernel at 2 when multiple conv layers are used:
        # after layer 1 collapses the feature dim, remaining spatial
        # size may be too small for larger kernels.
        # RNA_conv_kernel   = t.suggest_int("RNA_conv_kernel", 2, 2 if RNA_n_ConvLayer > 1 else 5)
        # DNA_conv_kernel   = t.suggest_int("DNA_conv_kernel", 2, 2 if DNA_n_ConvLayer > 1 else 5)
        if RNA_n_ConvLayer > 1:
            RNA_conv_kernel = 1
        else:
            RNA_conv_kernel = t.suggest_int("RNA_conv_kernel", 1, min_seq_len_rna)

        if DNA_n_ConvLayer > 1:
            DNA_conv_kernel = 1
        else:
            DNA_conv_kernel = t.suggest_int("DNA_conv_kernel", 1, min_seq_len_dna)
        last_ConvFClayer  = t.suggest_int("last_ConvFClayer", 1, 3)
        tcga_hidden_size  = t.suggest_categorical("tcga_hidden_size", [32, 64, 96, 128])
        ConvRelu          = t.suggest_categorical("ConvRelu", ["Yes", "No"])
        FullRelu          = t.suggest_categorical("FullRelu", ["Yes", "No"])
    else:
        d = trial_or_dict
        RNA_n_channel_1st = int(d["RNA_n_channel_1st"])
        DNA_n_channel_1st = int(d["DNA_n_channel_1st"])
        RNA_n_ConvLayer = int(d["RNA_n_ConvLayer"])
        DNA_n_ConvLayer = int(d["DNA_n_ConvLayer"])
        RNA_conv_kernel = int(d.get("RNA_conv_kernel", 1))
        DNA_conv_kernel = int(d.get("DNA_conv_kernel", 1))
        # RNA_conv_kernel = int(d["RNA_conv_kernel"])
        # DNA_conv_kernel = int(d["DNA_conv_kernel"])
        last_ConvFClayer  = int(d["last_ConvFClayer"])
        tcga_hidden_size  = int(d["tcga_hidden_size"])
        ConvRelu = d["ConvRelu"]
        FullRelu = d["FullRelu"]

    # ── Derived dimensions (must match forward() logic exactly) ───────────────
    # x_rna shape after RNA_layers:  (batch, RNA_n_channel_1st)  [after sum+flatten]
    # x_dna shape after DNA_layers:  (batch, DNA_n_channel_1st)  [after sum+flatten]
    # concatenated encode vector:    RNA_n_channel_1st + DNA_n_channel_1st
    encode_last_n_channel = RNA_n_channel_1st + DNA_n_channel_1st

    # concatenated final vector:     tcga_hidden_size + encode_last_n_channel
    last_n_channel = tcga_hidden_size + encode_last_n_channel

    return {
        **FIXED_PARAMS,
        "RNA_n_channel_1st":     RNA_n_channel_1st,
        "DNA_n_channel_1st":     DNA_n_channel_1st,
        "RNA_n_ConvLayer":       RNA_n_ConvLayer,
        "DNA_n_ConvLayer":       DNA_n_ConvLayer,
        "RNA_conv_kernel":       RNA_conv_kernel,
        "DNA_conv_kernel":       DNA_conv_kernel,
        "last_ConvFClayer":      last_ConvFClayer,
        "tcga_hidden_size":      tcga_hidden_size,
        "ConvRelu":              ConvRelu,
        "FullRelu":              FullRelu,
        # derived — do NOT tune independently
        "encode_last_n_channel": encode_last_n_channel,
        "last_n_channel":        last_n_channel,
    }


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    Sample hyperparameters, train for a short budget, return best val AUC-ROC.
    Optuna maximises this value.
    """
    # Training hyperparameters
    lr = trial.suggest_float("lr",     1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-1, log=True)
    step_size = trial.suggest_categorical("step_size", [10, 20, 30, 50])
    trial_epochs = trial.suggest_int("trial_epochs", 20, 60, step=10)

    # Architecture params (with correct derived dimensions)
    params = build_params(trial, is_trial=True)

    # Build batches
    train_steps, train_batches = batch_iter(
        tcga_train_data,
        encode_X_mRNA_train.values[:, 1],
        encode_X_promoter_train.values[:, 1],
        encode_Y_train.values,
        batch_size=batch_size, shuffle=True)

    val_steps, val_batches = batch_iter(
        tcga_val_data,
        encode_X_mRNA_val.values[:, 1],
        encode_X_promoter_val.values[:, 1],
        encode_Y_val.values,
        batch_size=batch_size, shuffle=False)

    # Build model
    try:
        model = ConcatedNet(params).to(device)
    except Exception as e:
        raise optuna.exceptions.TrialPruned(f"Model build failed: {e}")

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    best_pr_auc = float('-inf')
    # best_loss = float('inf')

    for epoch in range(trial_epochs):
        # Train
        model.train()
        data_iter = iter(train_batches)
        for _ in range(train_steps):
            X          = next(data_iter)
            x_mRNA     = X[0][0].view(X[0][0].shape[0], 1,
                                    X[0][0].shape[1], X[0][0].shape[2]).to(device)
            x_promoter = X[0][1].view(X[0][1].shape[0], 1,
                                    X[0][1].shape[1], X[0][1].shape[2]).to(device)
            x_tcga     = X[1].to(device)
            labels     = X[2].to(device)
            optimizer.zero_grad()
            try:
                outputs = model(x_mRNA, x_promoter, x_tcga)
            except RuntimeError as e:
                raise optuna.exceptions.TrialPruned(f"Forward pass failed: {e}")
            loss = criterion(outputs, labels) + model.l2_loss() * l2_reg
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        _, _, _, val_pr_auc, _, _, val_loss = evaluate(model, val_steps, val_batches, criterion, device)
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
        # if val_loss < best_loss:
        #     best_loss = val_loss

        # Report for pruning
        trial.report(val_pr_auc, epoch)
        # trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_pr_auc
    # return best_loss


# ── Run Optuna study ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Optuna hyperparameter search  ({N_TRIALS} trials)")
print(f"{'='*55}\n")

study = optuna.create_study(
    direction="maximize",
    # direction="minimize",
    sampler=TPESampler(seed=seed_value),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    study_name=f"concat_model_{tcga_cancer}",
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params    = study.best_params
best_val_pr_auc = study.best_value

print(f"\nBest val pr auc : {best_val_pr_auc:.5f}")
print("Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:28s}: {v}")

# ── Reconstruct full params with derived dimensions ───────────────────────────
# Separate training params from architecture params
training_keys = {"lr", "batch_size", "l2_reg", "step_size", "trial_epochs"}
best_arch_params     = {k: v for k, v in best_params.items() if k not in training_keys}
best_training_params = {k: v for k, v in best_params.items() if k in training_keys}

# Rebuild full params dict (recalculates derived dimensions correctly)
full_params = build_params(best_arch_params, is_trial=False)

print(f"\nDerived dimensions:")
print(f"  encode_last_n_channel = RNA({full_params['RNA_n_channel_1st']}) "
    f"+ DNA({full_params['DNA_n_channel_1st']}) "
    f"= {full_params['encode_last_n_channel']}")
print(f"  last_n_channel        = tcga_hidden({full_params['tcga_hidden_size']}) "
    f"+ encode_last({full_params['encode_last_n_channel']}) "
    f"= {full_params['last_n_channel']}")

# Save best architecture to params.json
with open(params_file, 'w') as f:
    json.dump(full_params, f, indent=4)
print(f"\nBest architecture params saved → {params_file}")

# Save full Optuna results
optuna_out = f"{outloc}/best_optuna_params_{tcga_cancer}.json"
with open(optuna_out, "w") as f:
    json.dump({
        "best_val_pr_auc":    best_val_pr_auc,
        "training":         best_training_params,
        "architecture":     best_arch_params,
        "derived":          {
        "encode_last_n_channel": full_params["encode_last_n_channel"],
        "last_n_channel":        full_params["last_n_channel"],
        }
    }, f, indent=2)
print(f"Full Optuna results saved → {optuna_out}")

# ── Diagnostic plots ──────────────────────────────────────────────────────────
plot_dir = f'../../plots_{tcga_cancer}/concat_model'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optuna optimisation history")
plt.tight_layout()
plt.savefig(f'{plot_dir}/optuna_history.png', dpi=300)
plt.close()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Hyperparameter importances")
plt.tight_layout()
plt.savefig(f'{plot_dir}/optuna_param_importance.png', dpi=300)
plt.close()

# ── Final retraining with best hyperparameters ────────────────────────────────
print(f"\n{'='*55}")
print(f"  Final retraining with best params ({FINAL_EPOCHS} epochs)")
print(f"{'='*55}\n")

best_batch_size = best_params["batch_size"]
best_lr         = best_params["lr"]
best_l2_reg     = best_params["l2_reg"]
best_step_size  = best_params["step_size"]

train_steps, train_batches = batch_iter(
    tcga_train_data,
    encode_X_mRNA_train.values[:, 1],
    encode_X_promoter_train.values[:, 1],
    encode_Y_train.values,
    batch_size=best_batch_size, shuffle=True)

val_steps, val_batches = batch_iter(
    tcga_val_data,
    encode_X_mRNA_val.values[:, 1],
    encode_X_promoter_val.values[:, 1],
    encode_Y_val.values,
    batch_size=best_batch_size, shuffle=False)

final_model = ConcatedNet(full_params).to(device)
# criterion = nn.CrossEntropyLoss()
criterion   = nn.CrossEntropyLoss(weight=class_weights.to(device))
# criterion = FocalLoss(alpha=class_weights.to(device),gamma=2)
optimizer   = optim.Adam(final_model.parameters(), lr=best_lr)
scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=best_step_size, gamma=0.1)

best_val_pr_auc = float('-inf')
# best_val_loss = float('inf')
train_losses = []
val_losses   = []
val_auc_rocs = []
val_accs     = []
val_f1s      = []
val_pr_aucs  = []

for epoch in range(FINAL_EPOCHS):
    # Train
    final_model.train()
    running_loss = 0.0
    data_iter    = iter(train_batches)

    for _ in range(train_steps):
        X          = next(data_iter)
        x_mRNA     = X[0][0].view(X[0][0].shape[0], 1,
                                X[0][0].shape[1], X[0][0].shape[2]).to(device)
        x_promoter = X[0][1].view(X[0][1].shape[0], 1,
                                X[0][1].shape[1], X[0][1].shape[2]).to(device)
        x_tcga     = X[1].to(device)
        labels     = X[2].to(device)
        optimizer.zero_grad()
        outputs  = final_model(x_mRNA, x_promoter, x_tcga)
        loss     = criterion(outputs, labels)
        l2_loss  = final_model.l2_loss() * best_l2_reg
        (loss + l2_loss).backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / train_steps
    train_losses.append(epoch_loss)

    # Validate
    val_auc_roc, val_accuracy, val_f1, val_pr_auc, val_recalls, val_cm, val_loss = evaluate(
        final_model, val_steps, val_batches, criterion, device)

    val_auc_rocs.append(val_auc_roc)
    val_accs.append(val_accuracy)
    val_f1s.append(val_f1)
    val_pr_aucs.append(val_pr_auc)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{FINAL_EPOCHS}: '
        f'train_loss={epoch_loss:.4f}  '
        f'AUC-ROC={val_auc_roc:.4f}  ACC={val_accuracy:.4f}  '
        f'F1={val_f1:.4f}  PR-AUC={val_pr_auc:.4f}')
    # 
    if val_pr_auc > best_val_pr_auc:
        best_val_pr_auc = val_pr_auc
        torch.save(final_model.state_dict(), f'{outloc}best_model.pth')

    torch.save(final_model.state_dict(), f'{outloc}epoch{epoch+1}.pth')

print(f'\nBest Val AUC-ROC : {max(val_auc_rocs):.4f}')
print(f'Best Val ACC     : {max(val_accs):.4f}')
print(f'Best Val F1      : {max(val_f1s):.4f}')
print(f'Best Val PR-AUC  : {max(val_pr_aucs):.4f}')

# Training performance
train_auc_roc, train_accuracy, train_f1, train_pr_auc, \
train_recalls, train_cm, train_loss = evaluate(
    final_model,
    train_steps,
    train_batches,
    criterion,
    device
)

print("\nTraining CM:")
print(train_cm)

print("Training recalls:")
print(train_recalls)

# ── Plotting ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses,   label='Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss',  fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/train_val_loss_final.png', dpi=300, bbox_inches="tight")
plt.close()

