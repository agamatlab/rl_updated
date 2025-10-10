#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import matplotlib
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
parser.add_argument('--path')
csv_path = Path()
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SMOOTH_WIN = 10
DPI = 300

csv_path = Path(args.path+ "/log.csv")
if not csv_path.exists():
    raise FileNotFoundError(f'Can not find {csv_path.resolve()}')

df = pd.read_csv(csv_path, sep=None, engine='python', skip_blank_lines=True)
df.columns = df.columns.str.strip().str.lower()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.ffill().bfill()

if 'frames' not in df.columns:
    raise KeyError('CSV column frames missing')

METRICS = {
    'rreturn_mean': 'Discounted Return',
    'policy_loss':  'Policy Loss',
    'value_loss':   'Value Loss',
    'entropy':      'Entropy'
}

def smooth(x):
    return x.rolling(SMOOTH_WIN, min_periods=1).mean()

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axes = axes.ravel()

for ax, (col, title) in zip(axes, METRICS.items()):
    if col not in df.columns:
        ax.text(0.5, 0.5, f'{col} missing', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
        ax.set_title(title)
        continue
    ax.plot(df['frames'], smooth(df[col]), color='tab:blue', lw=1.2)
    ax.set_title(title)
    ax.grid(alpha=0.3)

fig.supxlabel('Frames')
fig.supylabel('Smoothed Value')
fig.suptitle('Training Curves')
plt.tight_layout()

out_png = csv_path.with_name('training_curves.pdf')
plt.savefig(out_png, dpi=DPI)
print(f'Saved to -> {out_png}')

if args.show:
    plt.show()