
Example (Colab):
  !python train_fixed.py --data /content/drive/MyDrive/ecsmp_merged --dataset ECSMP \
       --model lstm --layers 3 --units 256 --window 60 --stride 10 --seeds 42
  !python train_fixed.py --data /content/drive/MyDrive/sweet_merged --dataset SWEET \
       --model transformer --layers 5 --units 256 --window 60 --stride 10 --seeds 42
Results are appended to results.jsonl; per-user EERs and class counts are saved as CSV.
"""
import argparse, glob, json, math, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--dataset', required=True, help='label written to results, e.g. ECSMP or SWEET')
p.add_argument('--model', choices=['lstm', 'transformer'], required=True)
p.add_argument('--layers', type=int, default=3, help='LSTM layers, or dense layers in the Transformer feed-forward block')
p.add_argument('--units', type=int, default=256)
p.add_argument('--window', type=int, default=60, help='window length in samples (= seconds at 1 Hz)')
p.add_argument('--stride', type=int, default=10, help='window stride in samples')
p.add_argument('--split', type=float, nargs=3, default=[0.70, 0.15, 0.15])
p.add_argument('--batch', type=int, default=64)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--patience', type=int, default=5)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--no_pos_enc', action='store_true', help='disable sinusoidal positional encoding (Transformer)')
p.add_argument('--seeds', type=int, nargs='+', default=[42])
p.add_argument('--out', default='results')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- 1. load, chronological split
files = sorted(glob.glob(os.path.join(a.data, '*.csv')))
blocks = {'train': [], 'val': [], 'test': []}          # list of (user_id, np.array rows)
feat_cols = None
for f in files:
    df = pd.read_csv(f)
    if feat_cols is None:
        feat_cols = list(df.columns[:-1])
    for uid, ud in df.groupby(df.columns[-1], sort=False):
        X = ud[feat_cols].to_numpy(dtype=np.float32)       # original row order kept
        n = len(X)
        c1 = int(a.split[0] * n); c2 = int((a.split[0] + a.split[1]) * n)
        blocks['train'].append((uid, X[:c1]))
        blocks['val'].append((uid, X[c1:c2]))
        blocks['test'].append((uid, X[c2:]))
users = sorted({u for u, _ in blocks['train']})
uid2idx = {u: i for i, u in enumerate(users)}
C = len(users)
print(f'{len(files)} files, {C} users, features: {feat_cols}')

# ---------------------------------------------------------------- 2. train-only imputation + scaling
train_all = np.concatenate([x for _, x in blocks['train']])
mu_imp = np.nanmean(train_all, axis=0)
missing_pct = {s: float(np.mean(np.isnan(np.concatenate([x for _, x in blocks[s]])))) * 100 for s in blocks}
def impute(x):
    x = x.copy(); r, c = np.where(np.isnan(x)); x[r, c] = mu_imp[c]; return x
train_imp = impute(train_all)
mu, sd = train_imp.mean(0), train_imp.std(0) + 1e-8
for s in blocks:
    blocks[s] = [(u, (impute(x) - mu) / sd) for u, x in blocks[s]]

# ---------------------------------------------------------------- 3. windows inside each block
def make_index(split):
    arrs, idx = [], []
    for u, x in blocks[split]:
        k = len(arrs); arrs.append(x)
        starts = np.arange(0, len(x) - a.window + 1, a.stride)
        idx.append(np.stack([np.full(len(starts), k), starts, np.full(len(starts), uid2idx[u])], 1))
    idx = np.concatenate(idx) if idx else np.zeros((0, 3), int)
    return arrs, idx

class WinSeq(tf.keras.utils.Sequence):
    def __init__(self, arrs, idx, shuffle, seed):
        super().__init__(); self.arrs, self.idx, self.shuffle = arrs, idx.copy(), shuffle
        self.rng = np.random.default_rng(seed); self.on_epoch_end()
    def __len__(self): return math.ceil(len(self.idx) / a.batch)
    def __getitem__(self, i):
        b = self.idx[i * a.batch:(i + 1) * a.batch]
        X = np.stack([self.arrs[k][s:s + a.window] for k, s, _ in b]).astype(np.float32)
        return X, b[:, 2]
    def on_epoch_end(self):
        if self.shuffle: self.rng.shuffle(self.idx)

# ---------------------------------------------------------------- 4. models
def build_lstm(F):
    m = tf.keras.Sequential([tf.keras.Input((a.window, F))])
    for i in range(a.layers):
        m.add(tf.keras.layers.LSTM(a.units, return_sequences=i < a.layers - 1))
    m.add(tf.keras.layers.Dense(16, activation='relu'))
    m.add(tf.keras.layers.Dense(C, activation='softmax'))
    return m

def pos_encoding(T, F):
    pos = np.arange(T)[:, None]; i = np.arange(F)[None, :]
    ang = pos / np.power(10000, (2 * (i // 2)) / F)
    pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
    return tf.constant(pe[None], dtype=tf.float32)

def build_transformer(F):
    L = tf.keras.layers
    inp = tf.keras.Input((a.window, F)); x = inp
    if not a.no_pos_enc:
        x = x + pos_encoding(a.window, F)
    att = L.MultiHeadAttention(num_heads=4, key_dim=F, dropout=a.dropout)(x, x)
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, att]))
    ff = x
    for _ in range(a.layers):
        ff = L.Dense(a.units, activation='relu')(ff)
    ff = L.Dropout(a.dropout)(L.Dense(F)(ff))
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, ff]))
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(16, activation='relu')(x)
    return tf.keras.Model(inp, L.Dense(C, activation='softmax')(x))

# ---------------------------------------------------------------- 5. EER (same definition as original code)
def eer_per_class(y, S):
    out = []
    for c in range(C):
        yb = (y == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): out.append(np.nan); continue
        fpr, tpr, _ = roc_curve(yb, S[:, c]); fnr = 1 - tpr
        j = np.argmin(np.abs(fpr - fnr)); out.append((fpr[j] + fnr[j]) / 2)
    return np.array(out)

# ---------------------------------------------------------------- 6. run
tr_arrs, tr_idx = make_index('train'); va_arrs, va_idx = make_index('val'); te_arrs, te_idx = make_index('test')
counts = pd.DataFrame({'user': users,
                       'train_windows_before': np.bincount(tr_idx[:, 2], minlength=C),
                       'val_windows': np.bincount(va_idx[:, 2], minlength=C),
                       'test_windows': np.bincount(te_idx[:, 2], minlength=C)})
# random oversampling of TRAIN windows only
target = counts.train_windows_before.max()
rng0 = np.random.default_rng(0); parts = []
for c in range(C):
    rows = tr_idx[tr_idx[:, 2] == c]
    if len(rows) == 0: continue
    extra = rows[rng0.integers(0, len(rows), target - len(rows))] if len(rows) < target else rows[:0]
    parts += [rows, extra]
tr_idx_os = np.concatenate(parts)
counts['train_windows_after'] = np.bincount(tr_idx_os[:, 2], minlength=C)
tag = f'{a.dataset}_{a.model}_L{a.layers}_U{a.units}_W{a.window}_S{a.stride}'
counts.to_csv(os.path.join(a.out, f'class_counts_{tag}.csv'), index=False)

for seed in a.seeds:
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    F = len(feat_cols)
    model = build_lstm(F) if a.model == 'lstm' else build_transformer(F)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    h = model.fit(WinSeq(tr_arrs, tr_idx_os, True, seed), validation_data=WinSeq(va_arrs, va_idx, False, seed),
                  epochs=a.epochs, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=a.patience, restore_best_weights=True)])
    best = int(np.argmin(h.history['val_loss']))
    S = model.predict(WinSeq(te_arrs, te_idx, False, seed), verbose=0)
    y = te_idx[:, 2]; yp = S.argmax(1)
    eers = eer_per_class(y, S)
    pd.DataFrame({'user': users, 'eer': eers}).to_csv(os.path.join(a.out, f'eer_per_user_{tag}_seed{seed}.csv'), index=False)
    res = dict(dataset=a.dataset, model=a.model, layers=a.layers, units=a.units, window=a.window, stride=a.stride,
               seed=seed, n_users=C, features=feat_cols, epochs_run=len(h.history['loss']), best_epoch=best + 1,
               train_loss=float(h.history['loss'][best]), train_acc=float(h.history['accuracy'][best]),
               val_acc=float(h.history['val_accuracy'][best]),
               test_acc=float((yp == y).mean()), test_macro_f1=float(f1_score(y, yp, average='macro')),
               eer_mean=float(np.nanmean(eers)), eer_sd=float(np.nanstd(eers)), eer_max=float(np.nanmax(eers)),
               missing_pct=missing_pct, n_windows=dict(train_before=int(len(tr_idx)), train_after=int(len(tr_idx_os)),
                                                       val=int(len(va_idx)), test=int(len(te_idx))),
               hyper=dict(batch=a.batch, lr=a.lr, patience=a.patience, dropout=a.dropout,
                          pos_enc=(a.model == 'transformer' and not a.no_pos_enc)),
               minutes=round((time.time() - t0) / 60, 1))
    print(json.dumps(res, indent=1))
    with open(os.path.join(a.out, 'results.jsonl'), 'a') as fh:
        fh.write(json.dumps(res) + '\n')
