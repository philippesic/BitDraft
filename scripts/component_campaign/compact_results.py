"""Keep compact findings in Git while retaining captures/weights/raw panels outside it."""
import json

from .common import R, ROOT, pin, save


def main():
    publication = json.loads((R / 'final-publication.json').read_text())
    if publication['status'] != 'MEASURED_PUBLICATION_READY_FOR_REVIEW':
        raise ValueError('Actual full publication must precede compact findings')
    rows = json.loads((R / 'charts/comparison.json').read_text())['rows']
    fields = (
        'arm', 'updates', 'component_spec', 'trainable_policy', 'model_sha256',
        'data_manifest_sha256', 'source_model_sha256', 'B0_checkpoint_sha256',
        'target_sha256', 'ce', 'agreement', 'mean_rank', 'mean_margin',
        'matched_recovery_control', 'ce_gap_matched_control',
        'supervised_positions', 'unique_supervised_positions_exposed',
        'repeated_supervised_positions', 'admitted_train_label_pool',
        'training_update_seconds', 'final_trainer_process_wall_seconds',
        'training_peak_allocator_bytes', 'training_peak_reserved_bytes',
        'training_peak_host_rss_bytes', 'tensor_bytes',
        'private_vocabulary_tensor_bytes', 'on_disk_duplicate_private_vocabulary_bytes',
        'combined_model_tensor_bytes', 'draft_reused_target_vocabulary_bytes',
        'actual_gpu_model_bytes', 'actual_cpu_model_bytes', 'private_gpu_model_bytes',
        'draft_cache_bytes', 'draft_rotation_bytes',
        'model_cache_payload_bytes', 'combined_model_and_draft_cache_payload_bytes',
        'sampled_combined_device_peak_bytes', 'clean_gpu_forward_ms',
        'full_draft_host_ms', 'full_draft_all_phases_host_ms_per_round',
        'selection_request_tok_s', 'selection_accepted', 'selection_proposed',
        'selection_acceptance', 'final_request_tok_s', 'final_accepted',
        'final_proposed', 'final_acceptance', 'final_prefix_counts',
        'final_emitted_per_round', 'final_vs_q4', 'final_request_vs_matched_control',
        'final_request_cohort', 'final_request_repetitions',
        'full_recovery_state_status', 'frontier_full_state', 'status')
    latest = []
    for name in dict.fromkeys(r['arm'] for r in rows):
        row = max((r for r in rows if r['arm'] == name), key=lambda r: r['updates'])
        latest.append({key: row.get(key) for key in fields})
    accounting = json.loads((R / 'accounting.json').read_text())
    destination = ROOT / 'experiments/component-campaign-20261105/results.json'
    save(destination, dict(
        status='COMPACT_MEASURED_FINDINGS', endpoints=latest,
        physical_recovery_updates=accounting['physical_updates_completed'],
        qualification_optimizer_updates=accounting['qualification_optimizer_updates'],
        cumulative_actual_update_seconds=accounting['cumulative_actual_update_seconds'],
        checkpoint_save_seconds=accounting['checkpoint_save_seconds'],
        recipe_decision=json.loads((R / 'final-recipe-decision.json').read_text()),
        publication=pin(R / 'final-publication.json'),
        full_comparison=pin(R / 'charts/comparison.json'),
        all_separate_panels=pin(R / 'charts/all-panels.json'),
        interactive=pin(R / 'charts/comparison.html'),
        frontier_recovery=pin(R / 'qualification/frontier-recovery/result.json'),
        source=pin(__file__),
        limitations=publication['limits'],
        retention='Weights, native captures, raw results, source closures and full charts remain at their authenticated canonical campaign paths outside Git. These path/hash records do not claim an independent artifact backup. Local source/results commits remain subject to the explicit no-push instruction.'))
    print('COMPACT_MEASURED_FINDINGS_SAVED', destination, flush=True)


if __name__ == '__main__':
    main()
