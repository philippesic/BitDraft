"""Additional measured recovery, exposure, prefix and memory views for publication."""
import json

from .common import R, pin, save


def generate(summary, comparison, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    rows = comparison['rows']
    names = list(dict.fromkeys(r['arm'] for r in rows))
    latest = {n: max((r for r in rows if r['arm'] == n), key=lambda r: r['updates']) for n in names}
    groups = [
        ('Head', ['H0', 'H4', 'H3', 'H4R2', 'H6R2']),
        ('Embedding / tied', ['H0', 'E4', 'HE4', 'HE4R2', 'HE6R2']),
        ('Draft K/V', ['H0', 'K8F', 'KF8', 'K88', 'K48', 'K84', 'K44']),
        ('Scalar storage', ['H0', 'Sscale', 'Smidpoint', 'Sgain', 'Sclip', 'Snorm', 'Sall']),
        ('Combinations', ['H0', 'Csmall', 'Chead', 'CtieKV', 'Ctied']),
        ('Broader TRAIN recipe', ['H0broad', 'Csmallbroad'])]
    artifacts = []

    def finish(fig, name):
        fig.tight_layout()
        for suffix in ('.png', '.svg'):
            path = out / (name + suffix)
            fig.savefig(path, dpi=145)
            artifacts.append(pin(path))
        plt.close(fig)

    def history(field, label, name, logarithmic=False):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (title, members) in zip(axes.flat, groups):
            for member in members:
                points = sorted((r for r in rows if r['arm'] == member and r.get(field) is not None), key=lambda r: r['updates'])
                if points:
                    ax.plot([r['updates'] for r in points], [r[field] for r in points], marker='o', markersize=3, label=member)
            if logarithmic:
                ax.set_yscale('log')
            ax.set(title=title, xlabel='Additional matched updates', ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2)
        fig.suptitle('Consumed DEV · original and broader TRAIN recipes remain separate · one training history per recipe')
        finish(fig, name)

    history('mean_rank', 'Mean target-token rank · lower is better', 'recovery-target-rank', True)
    history('mean_margin', 'Target logit minus strongest competitor', 'recovery-target-margin')
    history('agreement', 'Full-vocabulary top1 agreement', 'recovery-target-agreement')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, members) in zip(axes.flat, groups):
        for member in members:
            points = sorted((r for r in rows if r['arm'] == member and r.get('ce_gap_matched_control') is not None), key=lambda r: r['updates'])
            if points:
                ax.plot([r['training_update_seconds'] / 60 for r in points], [r['ce_gap_matched_control'] for r in points], marker='o', markersize=3, label=member)
        ax.set(title=title, xlabel='Measured update compute · minutes', ylabel='DEV CE gap to matched control')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle('Measured recovery compute · setup, DEV, export, checkpoint and native evaluation excluded')
    finish(fig, 'recovery-compute-time')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, row in latest.items():
        wall, gap = row.get('final_trainer_process_wall_seconds'), row.get('ce_gap_matched_control')
        if wall is not None and gap is not None:
            axes[0].scatter(wall / 60, gap)
            axes[0].annotate(name, (wall / 60, gap), fontsize=7)
        points = sorted((r for r in rows if r['arm'] == name and r.get('readout_elapsed_trainer_wall_seconds') is not None and r.get('ce_gap_matched_control') is not None), key=lambda r: r['updates'])
        if points:
            axes[1].plot([r['readout_elapsed_trainer_wall_seconds'] / 60 for r in points], [r['ce_gap_matched_control'] for r in points], marker='o', markersize=3, label=name)
    axes[0].set(title='Final4096 endpoints · all recorded trainer processes', xlabel='Full trainer wall · minutes', ylabel='DEV CE gap to matched control')
    axes[1].set(title='Intermediate elapsed wall · only where actually recorded', xlabel='Elapsed trainer wall · minutes', ylabel='DEV CE gap to matched control')
    axes[1].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Actual trainer wall includes setup, DEV, exports and checkpoint I/O · separate native panels / captures excluded')
    finish(fig, 'recovery-actual-wall-time')

    final = summary['native']['final-heldout24']['arms']
    selected = [n for n in final if n != 'target']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(selected))
    for index in (0, 1):
        axes[0].bar(x + (index - .5) * .36, [final[n]['prefix_counts'][index] / final[n]['rounds'] for n in selected], width=.36, label='Survives draft prefix ' + str(index + 1))
    ratios = [final[n]['vs_q4']['ratio'] for n in selected]
    intervals = [final[n]['vs_q4']['paired_95_interval'] for n in selected]
    axes[1].bar(x, ratios)
    axes[1].errorbar(x, ratios, yerr=[[max(0, r - q[0]) for r, q in zip(ratios, intervals)], [max(0, q[1] - r) for r, q in zip(ratios, intervals)]], fmt='none', color='black', capsize=3)
    axes[1].axhline(1, linestyle='--', color='gray')
    axes[0].set(ylabel='Prefix survival count / speculative rounds', title='Actual accepted prefix history')
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Full request speed / external Q4', title='Paired95% prompt/process interval')
    for ax in axes:
        ax.set_xticks(x, selected, rotation=45, ha='right')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Locked fresh24 heldout · 2 warmups +5 alternating repetitions · intervals exclude training-seed uncertainty')
    finish(fig, 'formal-prefix-and-request-uncertainty')

    selected = [n for n in selected if n in latest]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    parts = [('private_vocabulary_tensor_bytes', 'Private head / embedding'), ('other_draft_tensor_bytes', 'Other private draft tensors'), ('draft_cache_bytes', 'Allocated draft K/V'), ('draft_rotation_bytes', 'Draft cache rotations')]
    for ax, target in ((axes[0], False), (axes[1], True)):
        bottom = np.zeros(len(selected))
        if target:
            target_bytes = np.array([latest[n]['immutable_target_tensor_bytes'] for n in selected]) / 2**20
            ax.bar(selected, target_bytes, label='Immutable target tensors')
            bottom += target_bytes
        for key, label in parts:
            vals = [latest[n]['tensor_bytes'] - latest[n].get('private_vocabulary_tensor_bytes', 0) if key == 'other_draft_tensor_bytes' else latest[n].get(key, 0) or 0 for n in selected]
            vals = np.array(vals) / 2**20
            ax.bar(selected, vals, bottom=bottom, label=label)
            bottom += vals
        ax.set(ylabel='MiB', title='Combined payload' if target else 'Private draft payload +allocated cache')
        ax.legend(fontsize=7)
    positions = np.arange(len(selected))
    for i, (key, label) in enumerate([('training_peak_allocator_bytes', 'Torch allocated'), ('training_peak_reserved_bytes', 'Torch reserved'), ('training_peak_host_rss_bytes', 'Host RSS')]):
        axes[2].bar(positions + (i - 1) * .25, [latest[n][key] / 2**30 if latest[n].get(key) is not None else np.nan for n in selected], width=.25, label=label)
    axes[2].set_xticks(positions, selected)
    axes[2].set(ylabel='GiB', title='Measured training peaks · references untrained')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Payload is not device residency · target cache / compute workspace / desktop are separate · shared vocabulary counted once')
    finish(fig, 'formal-memory-and-training-peaks')
    save(out / 'extended-chart-index.json', dict(figures=artifacts, source=pin(__file__), comparison=pin(out / 'comparison.json'), summary=pin(R / 'final-summary.json')))


if __name__ == '__main__':
    generate(json.loads((R / 'final-summary.json').read_text()), json.loads((R / 'charts/comparison.json').read_text()), R / 'charts')
