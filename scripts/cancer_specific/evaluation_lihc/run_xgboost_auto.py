import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, average_precision_score, recall_score, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
from utils.data_tool import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Training")
parser.add_argument('tcga_cancer',      type=str)
parser.add_argument('encode_cell_line', type=str)
parser.add_argument('outloc',           type=str)
parser.add_argument('--n_trials', type=int, default=50,
                    help='Number of Optuna trials (default: 50)')
args = parser.parse_args()

tcga_cancer      = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc           = args.outloc
N_TRIALS         = args.n_trials

# ── Data loading ─────────────────────────────────────────────────────────────
data = pd.read_csv(f'../../results/TCGAprocessed/{tcga_cancer}/merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)
data = data.drop(columns=['DElabel'], axis=1)

mRNA_data_loc     = '../../results/rna_features/'
promoter_data_loc = '../../results/promoter_features/'

# ── Data split ───────────────────────────────────────────────────────────────
train_names = f'../../data/modeling/{tcga_cancer}/full_train.csv'
val_names   = f'../../data/modeling/{tcga_cancer}/full_val.csv'
test_names  = f'../../data/modeling/{tcga_cancer}/full_test.csv'

train = pd.read_csv(train_names, sep="\t", header=0).values[:, 0]
val   = pd.read_csv(val_names,   sep="\t", header=0).values[:, 0]
test  = pd.read_csv(test_names,  sep="\t", header=0).values[:, 0]

# ── Preprocessing ─────────────────────────────────────────────────────────────
tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
tcga_val_df   = data.loc[data['Gene'].isin(val)].set_index('Gene')
tcga_test_df  = data.loc[data['Gene'].isin(test)].set_index('Gene')

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


def flatten_feature(df):
    """Sum binding nucleotides across positions (removes location dimension)."""
    features = []
    for mat in df.values[:, 1]:
        mat = mat.toarray() if hasattr(mat, "toarray") else np.array(mat)
        features.append(mat.sum(axis=1))
    return np.vstack(features)


# TCGA arrays
X_train_tcga = tcga_train_df.iloc[:, :-1].values
X_val_tcga   = tcga_val_df.iloc[:,   :-1].values
X_test_tcga  = tcga_test_df.iloc[:,  :-1].values

y_train = tcga_train_df.iloc[:, -1].values
y_val   = tcga_val_df.iloc[:,   -1].values
y_test  = tcga_test_df.iloc[:,  -1].values

# ENCODE arrays
mRNA_train = flatten_feature(encode_X_mRNA_train)
mRNA_val   = flatten_feature(encode_X_mRNA_val)
mRNA_test  = flatten_feature(encode_X_mRNA_test)

promoter_train = flatten_feature(encode_X_promoter_train)
promoter_val   = flatten_feature(encode_X_promoter_val)
promoter_test  = flatten_feature(encode_X_promoter_test)

# Final concatenated feature matrices
X_train = np.concatenate([X_train_tcga, mRNA_train, promoter_train], axis=1)
X_val   = np.concatenate([X_val_tcga,   mRNA_val,   promoter_val],   axis=1)
X_test  = np.concatenate([X_test_tcga,  mRNA_test,  promoter_test],  axis=1)

sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),

        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 保存最佳迭代轮数
    trial.set_user_attr("best_iteration", model.best_iteration)

    # ---------- 用最佳树数预测 Validation ----------
    val_probs = model.predict_proba(
        X_val,
        iteration_range=(0, model.best_iteration + 1)
    )

    y_val_onehot = np.eye(3)[y_val.astype(int)]

    val_pr_auc = average_precision_score(
        y_val_onehot,
        val_probs,
        average="macro"
    )

    return val_pr_auc

# ── Run Optuna study ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Optuna hyperparameter search  ({N_TRIALS} trials)")
print(f"{'='*55}\n")


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name=f"xgboost_{tcga_cancer}",
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_val_score = study.best_value

# print(f"\nBest validation log-loss : {best_val_score:.5f}")
print(f"\nBest validation Macro PR-AUC : {best_val_score:.5f}")
print("Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:25s}: {v}")

# Save best params to JSON
params_out = f"{outloc}/best_params_{tcga_cancer}.json"
'''
with open(params_out, "w") as f:
    json.dump({"best_val_logloss": best_val_score, **best_params}, f, indent=2)
'''
with open(params_out, "w") as f:
    json.dump(
        {
            "best_val_pr_auc": best_val_score,
            **best_params,
        },
        f,
        indent=2,
    )
print(f"\nBest params saved → {params_out}")

# ── Diagnostic plots ──────────────────────────────────────────────────────────
plot_dir = f'../../plots_{tcga_cancer}'

# 1. Optimisation history
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optuna optimisation history")
plt.tight_layout()
plt.savefig(f'{plot_dir}/optuna_history.png', dpi=300)
plt.close()

# 2. Hyperparameter importances
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Hyperparameter importances")
plt.tight_layout()
plt.savefig(f'{plot_dir}/optuna_param_importance.png', dpi=300)
plt.close()

# ── Retrain best model on train+val ──────────────────────────────────────────
print("\nRetraining best model on train + val …")

X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval  = np.concatenate([y_train,  y_val],  axis=0)
sw_trainval = compute_sample_weight(class_weight='balanced', y=y_trainval)

best_iteration = study.best_trial.user_attrs["best_iteration"]

# best_params already contains n_estimators from Optuna's search space,
# so exclude it before unpacking to avoid passing it twice.
final_params = {k: v for k, v in best_params.items() if k != 'n_estimators'}

final_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    tree_method='hist',
    random_state=42,
    n_estimators=best_iteration + 1,   # fixed — no early stopping on retrain
    **final_params,
)
final_model.fit(X_trainval, y_trainval, sample_weight=sw_trainval, verbose=False)

# ── Evaluation ────────────────────────────────────────────────────────────────
# ===== Training (train + val) evaluation =====
y_trainval_probs = final_model.predict_proba(X_trainval)
y_trainval_preds = np.argmax(y_trainval_probs, axis=1)

train_recalls = recall_score(
    y_trainval,
    y_trainval_preds,
    average=None
)

train_cm = confusion_matrix(y_trainval, y_trainval_preds)

# ===== Test evaluation =====
# ── Evaluation ────────────────────────────────────────────────────────────────

# ===== Training (train + val) evaluation =====
y_trainval_probs = final_model.predict_proba(X_trainval)
y_trainval_preds = np.argmax(y_trainval_probs, axis=1)

train_recalls = recall_score(
    y_trainval,
    y_trainval_preds,
    average=None
)

train_cm = confusion_matrix(y_trainval, y_trainval_preds)


# ===== Test evaluation =====
y_probs = final_model.predict_proba(X_test)
y_preds = np.argmax(y_probs, axis=1)

# One-hot encode y_test for PR-AUC
num_classes = 3
y_test_onehot = np.eye(num_classes)[y_test.astype(int)]

auc_roc = roc_auc_score(
    y_test, y_probs,
    average='macro',
    multi_class='ovr'
)

accuracy = accuracy_score(y_test, y_preds)

f1 = f1_score(
    y_test,
    y_preds,
    average='macro',
    zero_division=np.nan
)

pr_auc = average_precision_score(
    y_test_onehot,
    y_probs,
    average='macro'
)

recalls = recall_score(
    y_test,
    y_preds,
    average=None
)

cm = confusion_matrix(y_test, y_preds)


print("\n===== FINAL TRAINING RESULTS =====")
print(f"TRAINING RECALLS = {train_recalls}")
print("TRAINING confusion matrix:")
print(train_cm)

print("\n===== FINAL TEST RESULTS =====")
print(f"AUC-ROC   : {auc_roc:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"F1        : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"TEST RECALLS = {recalls}")
print("TEST confusion matrix:")
print(cm)