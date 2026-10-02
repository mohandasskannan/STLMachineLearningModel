"""Collect selected checkpoint metrics and saved generation diagnostics."""
import json
from pathlib import Path
from statistics import mean

import numpy as np

from stlmodel.common import file_hash, load_checkpoint

root = Path('outputs/procedural-v1')
result = {}
for kind in ('vae', 'diffusion'):
    run = Path(f'runs/procedural-{kind}-v1')
    state = load_checkpoint(run / 'best.pt', kind)
    events = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    selected = next(e for e in events if e['step'] == state['step'])
    result[kind] = {
        'checkpoint': str(run / 'best.pt'), 'sha256': file_hash(run / 'best.pt'),
        'selected_step': state['step'], 'selected_epoch': state['epoch'],
        'selected_validation': selected['validation'],
        'run_summary': json.loads((run / 'summary.json').read_text()),
        'test': json.loads((root / f'{kind}-test.json').read_text())['metrics'],
    }
cache = load_checkpoint('data/procedural-embeddings-v1.pt', 'embeddings')
result['cache_matches_selected_vae'] = cache['vae_sha256'] == result['vae']['sha256']
result['diffusion_matches_selected_vae'] = load_checkpoint('runs/procedural-diffusion-v1/best.pt', 'diffusion')['vae_sha256'] == result['vae']['sha256']
result['cache_examples'] = {s: len(cache[s]['ids']) for s in ('train', 'val', 'test')}
result['text_revision'] = cache['text_revision']
evaluation = json.loads((root / 'comparisons/evaluation.json').read_text())
rows = evaluation['samples']
result['generated_meshes'] = {
    'count': len(rows), 'failures': sum('error' in r for r in rows),
    'watertight': sum(r.get('mesh', {}).get('watertight', False) for r in rows),
    'components': [r.get('mesh', {}).get('components') for r in rows],
    'boundary_capped': sum(any('boundary' in w for w in r.get('mesh', {}).get('warnings', [])) for r in rows),
    'nearest_train_iou': [r['nearest_train_iou'] for r in rows],
}
result['prompt_pair_iou'] = []
for p, q in ((0, 1), (0, 2), (1, 2)):
    comparisons = [c for c in evaluation['comparisons'] if c['a'] // 3 == p and c['b'] // 3 == q and rows[c['a']]['seed'] == rows[c['b']]['seed']]
    result['prompt_pair_iou'].append({'prompts': [rows[3*p]['prompt'], rows[3*q]['prompt']],
                                    'by_seed': [c['voxel_iou'] for c in comparisons],
                                    'mean': mean(c['voxel_iou'] for c in comparisons)})
result['seed_diversity_iou'] = []
for p in range(3):
    values = [c['voxel_iou'] for c in evaluation['comparisons'] if c['a'] // 3 == p and c['b'] // 3 == p]
    result['seed_diversity_iou'].append({'prompt': rows[3*p]['prompt'], 'pairwise_iou': values, 'mean': mean(values)})
checks = json.loads((root / 'comparisons/reopened-mesh-checks.json').read_text())
result['reopened_all_finite'] = all(c['finite'] for c in checks)
result['reopened_all_longest_dimension_100mm'] = all(np.isclose(max(c['extents_mm']), 100, atol=1e-4) for c in checks)
(root / 'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
