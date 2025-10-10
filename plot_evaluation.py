#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import matplotlib


WIN = 20
DPI = 300

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
parser.add_argument('--path')
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = Path(args.path+ "/eval_logs/logs.csv")
if not csv_path.exists():
    raise FileNotFoundError(f'Can not find {csv_path.resolve()}')

df = pd.read_csv(csv_path, sep=None, engine='python', skip_blank_lines=True)
df.columns = df.columns.str.strip().str.lower()

for must in ['episode', 'return']:
    if must not in df.columns:
        raise KeyError(f'CSV column missing: {must}')

df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill()

df['return_smooth'] = df['return'].rolling(window=WIN, min_periods=1).mean()

plt.figure(figsize=(8, 5))
plt.plot(df['episode'], df['return'], alpha=0.3, label='raw return')
plt.plot(df['episode'], df['return_smooth'], color='tab:blue', lw=2,
         label=f'smooth (win={WIN})')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Evaluation Return')
plt.grid(alpha=0.3)
plt.legend()

out_png = csv_path.with_name('eval_return.pdf')
plt.savefig(out_png, dpi=DPI)
print(f'Saved to -> {out_png}')

if args.show:
    plt.show()

    
'''
eval_log_dir = os.path.join(model_dir, "eval_logs")
os.makedirs(eval_log_dir, exist_ok=True)
eval_log_file = os.path.join(eval_log_dir, f"logs.csv")

with open(eval_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "return", "num_frames"])
    for i in range(len(logs["return_per_episode"])):
        writer.writerow([i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]])

print(f"Evaluation results saved to {eval_log_file}")
'''