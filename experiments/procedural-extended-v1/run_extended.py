"""Resume the existing diffusion experiment, then save self-contained review artifacts."""
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from torch.utils.data import TensorDataset

from stlmodel.common import file_hash, load_checkpoint, write_json
from stlmodel.models import Denoiser, Diffusion
from stlmodel.pipeline import Generator, evaluate_generation
from stlmodel.training import evaluate_diffusion

ROOT = Path('outputs/procedural-extended-v1')
RUN = Path('runs/procedural-diffusion-extended-v1')
OLD = Path('runs/procedural-diffusion-v1')
VAE = Path('runs/procedural-vae-v1/best.pt')
CACHE = Path('data/procedural-embeddings-v1.pt')
PROMPTS = ['A chair with a tall back and armrests', 'a chair with armrests',
           'a chair without armrests', 'a table with four legs']


def status(stage, **extra):
    value = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **extra}
    write_json(ROOT / 'status.json', value)
    print(json.dumps(value), flush=True)


def assess(checkpoint):
    state = load_checkpoint(checkpoint, 'diffusion')
    cache = load_checkpoint(CACHE, 'embeddings')
    model = Denoiser(**state['architecture']).to('cuda')
    model.load_state_dict(state['model'])
    schedule = Diffusion(state['diffusion_steps'], 'cuda')
    values = {split: evaluate_diffusion(model, TensorDataset(cache[split]['latents'], cache[split]['text']), schedule, torch.device('cuda')) for split in ('val', 'test')}
    return {'checkpoint': str(checkpoint), 'sha256': file_hash(checkpoint),
            'step': state['step'], 'epoch': state['epoch'], 'metrics': values}


def comparisons(checkpoint, name, encoder=None):
    generator = Generator(VAE, checkpoint, 'cuda', encoder=encoder)
    report = evaluate_generation(generator, Path('data/fixture/manifest.json'), PROMPTS, ROOT / name)
    for path in sorted((ROOT / name).glob('*.stl')):
        # Exact generation sidecars record the prompt, seed and paired model hashes.
        p, seed = int(path.stem.split('-')[1]), int(path.stem.split('-')[3])
        write_json(path.with_suffix('.generation.json'), {**generator.metadata, 'prompt': PROMPTS[p], 'seed': seed, 'sampling_steps': 50, 'threshold': 0.5, 'size_mm': 100})
    subprocess.run([sys.executable, 'outputs/procedural-v1/render_review.py', str(ROOT / name), '--columns', '3', '--front'], check=True, stdout=subprocess.DEVNULL)
    return report, generator.encoder


def geometry_stats(report):
    samples = report['samples']
    return {'exported': sum('mesh' in r for r in samples), 'errors': [r for r in samples if 'error' in r],
            'components': [r.get('mesh', {}).get('components') for r in samples],
            'watertight': sum(r.get('mesh', {}).get('watertight', False) for r in samples),
            'boundary_capped': sum(any('boundary' in w for w in r.get('mesh', {}).get('warnings', [])) for r in samples),
            'nearest_train_iou': [r['nearest_train_iou'] for r in samples]}


def main():
    if RUN.exists():
        raise RuntimeError(f'{RUN} already exists; inspect its checkpoints before resuming')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    status('baseline', gpu=torch.cuda.get_device_name(), training_budget_seconds=3600)
    baseline = assess(OLD / 'best.pt')
    before, encoder = comparisons(OLD / 'best.pt', 'before')
    vae_hash = file_hash(VAE)
    cache_hash = file_hash(CACHE)
    RUN.mkdir(parents=True)
    for name in ('last.pt', 'best.pt', 'metrics.jsonl'):
        shutil.copy2(OLD / name, RUN / name)
    status('training', run=str(RUN), initial_step=3000, training_budget_seconds=3600)
    command = [sys.executable, '-m', 'stlmodel', 'train-diffusion', '--cache', str(CACHE),
               '--out', str(RUN), '--resume', str(RUN / 'last.pt'), '--device', 'cuda',
               '--epochs', '1000000', '--max-hours', '1']
    write_json(ROOT / 'command.json', {'command': command, 'readme_commit': 'ea35eb6', 'prompts': PROMPTS, 'seeds': [42, 43, 44], 'vae_sha256': vae_hash, 'cache_sha256': cache_hash})
    with (ROOT / 'training-console.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    status('evaluating')
    if file_hash(VAE) != vae_hash or file_hash(CACHE) != cache_hash:
        raise RuntimeError('Frozen VAE or embedding cache changed during training')
    after_metrics = assess(RUN / 'best.pt')
    after, _ = comparisons(RUN / 'best.pt', 'after', encoder)
    checks = json.loads((ROOT / 'after/reopened-mesh-checks.json').read_text())
    result = {'gpu': torch.cuda.get_device_name(), 'fixed_vae_sha256': vae_hash, 'cache_sha256': cache_hash,
              'baseline': baseline, 'extended': after_metrics, 'prompts': PROMPTS, 'seeds': [42, 43, 44],
              'training_summary': json.loads((RUN / 'summary.json').read_text()),
              'before_geometry': geometry_stats(before), 'after_geometry': geometry_stats(after),
              'reopened_all_finite': all(c['finite'] for c in checks),
              'reopened_scale_correct': all(np.isclose(max(c['extents_mm']), 100, atol=1e-4) for c in checks)}
    write_json(ROOT / 'results.json', result)
    status('complete', results=str(ROOT / 'results.json'), training_seconds=result['training_summary']['session_seconds'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise
