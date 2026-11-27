"""Render the recorded LR=0.01 five-seed tail-300 table, without rerunning FL."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'fedavg': 'FedAvg', 'poc': 'PoC', 'is_ht': 'PracIS-HT', 'delta_ht': 'PracDELTA-HT',
         'divfl': 'DivFL', 'hcs_static_gram_equal': 'HCSFed-static', 'uniform_real': 'Uniform HT',
         'score_real': 'Proxy-score HT', 'cluster_score_real_k3': 'Proxy-score + K=3',
         'representative_real_eta1p0': 'PJCS'}


def render(value):
    records = value['records']
    index = {(r['dataset'], r['alpha'], r['route']): r for r in records}
    config = json.loads((ROOT / 'configs/paper_lr001.json').read_text(encoding='utf-8'))
    expected = {(d, a, m) for d in config['dataset_names'] for a in config['alphas'] for m in config['routes']}
    if len(records) != 100 or set(index) != expected:
        raise ValueError('Expected the complete 100-group paper matrix')
    for row in records:
        if set(row['seeds']) != {str(seed) for seed in config['fl_seeds']}:
            raise ValueError('Five FL seeds are required')
        values = list(row['seeds'].values())
        if abs(statistics.mean(values) - row['mean_percent']) > 1e-10:
            raise ValueError('Recorded mean differs from seed values')
        if abs(statistics.stdev(values) - row['standard_deviation_pp']) > 1e-10:
            raise ValueError('Recorded sample SD differs from seed values')
    lines = ['Per seed: mean ACC over rounds 1201–1500; mean ± sample SD across five FL seeds.',
             '★ marks the highest unrounded mean within a dataset/alpha column; it is not a significance test.', '']
    for dataset in config['dataset_names']:
        lines += [dataset, '', '| Method | ' + ' | '.join(a.replace('p', '.') for a in config['alphas']) + ' |',
                  '| --- | ' + ' | '.join('---:' for _ in config['alphas']) + ' |']
        maxima = {a: max(index[dataset, a, m]['mean_percent'] for m in config['routes']) for a in config['alphas']}
        for route in config['routes']:
            cells = []
            for alpha in config['alphas']:
                row = index[dataset, alpha, route]
                text = f"{row['mean_percent']:.2f} ± {row['standard_deviation_pp']:.2f}"
                cells.append('**' + text + ' ★**' if row['mean_percent'] == maxima[alpha] else text)
            lines.append('| ' + NAMES[route] + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/lr001_tail300.json')
    args = parser.parse_args()
    print(render(json.loads(args.input.read_text(encoding='utf-8'))))
