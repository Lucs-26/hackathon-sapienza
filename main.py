import torch
import glob
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path

from utils import functions as uf
from utils.model import DynamicMLP

folder_path = './data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


csv_files = glob.glob(os.path.join(folder_path, '*c000.csv'))
df_all = pd.concat((pd.read_csv(file, sep=";") for file in csv_files), ignore_index=True)
    
random_seed = 42
train_df = df_all
id_col = "user_id"


# --- split into retain / validation / forget ---
forget_df = pd.read_csv(os.path.join(folder_path, 'forget_data.csv'))

df_all = df_all.drop_duplicates(subset=id_col, keep='first').reset_index(drop=True)

forget_ids = set(forget_df[id_col])
retain_df = df_all[~df_all[id_col].isin(forget_ids)].reset_index(drop=True)

# validation is carved out of the retain set only, never from Df
rng = np.random.default_rng(random_seed)
perm = rng.permutation(len(retain_df))
n_val = int(0.10 * len(retain_df))
val_df = retain_df.iloc[perm[:n_val]].reset_index(drop=True)
train_df = retain_df.iloc[perm[n_val:]].reset_index(drop=True)

assert set(val_df[id_col]).isdisjoint(forget_ids)
print(f"train {len(train_df)} | val {len(val_df)} | forget {len(forget_df)}")


X_train, y_train, feature_cols, target_cols = uf.prepare_data(train_df, id_col=id_col, target_prefix='target__')

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train).astype(np.float32)
# --- apply the same imputer to the other splits (fit on train only) ---
X_val, y_val, _, _ = uf.prepare_data(val_df, id_col=id_col, target_prefix='target__')
X_val = imputer.transform(X_val).astype(np.float32)

X_forget, y_forget, _, _ = uf.prepare_data(forget_df, id_col=id_col, target_prefix='target__')
X_forget = imputer.transform(X_forget).astype(np.float32)


pos_counts = np.sum(y_train, axis=0)
neg_counts = len(y_train) - pos_counts
pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), device=device)
pos_weights = pos_weights.clamp(min=0.1, max=100.0)
print(f"pos_weights: {pos_weights}")


artifact_path = Path('data') / 'model_artifact'

payload = uf.load_pickle(artifact_path)

state_dict = payload['state_dict']
architecture = payload['architecture']
best_params = payload['best_hyperparameters']
model_class_source = payload['model_class_source']

print("\n--- Saved Metadata ---")
print("Architecture parameters:", architecture)
print("Best Hyperparameters:", best_params)

try:
    model = DynamicMLP(
        input_dim=architecture['input_dim'],
        hidden_layers=architecture['hidden_layers'],
        num_outputs=architecture['num_outputs']
    )
except NameError:
    print("DynamicMLP class was not found. Check if the class source compiled correctly.")
    raise

model.load_state_dict(state_dict)

model.eval()

print("\nModel successfully reconstructed and weights loaded.")