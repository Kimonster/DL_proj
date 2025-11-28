import os
import json
import pandas as pd

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='README.md')
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, 'config.json')
    metrics_path = os.path.join(args.model_dir, 'metrics.csv')

    lines = []
    lines.append('# Training Summary\n')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
        lines.append('**Training configuration**:\n')
        for k, v in cfg.items():
            lines.append(f'- {k}: {v}')
        lines.append('\n')
    else:
        lines.append('No `config.json` found.\n')

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        lines.append('**Metrics (CSV)**:\n')
        lines.append(df.tail(20).to_markdown())
        lines.append('\n')
    else:
        lines.append('No metrics found.\n')

    with open(os.path.join(args.model_dir, args.out), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('README generated at', os.path.join(args.model_dir, args.out))
