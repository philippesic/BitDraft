import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_pytorch_w1a1 import encode_prompt, load_prompts, sha256_file
from w1a1_eagle.official_loader import load_official_eagle3
import angelslim.compressor.speculative.inference.models.eagle3.eagle3_model as eagle3_mod
from angelslim.compressor.speculative.utils.util import reset_tree_mode

config_path = ROOT / "configs/pytorch_w1a1.toml"
config = tomllib.loads(config_path.read_text())
evaluation = config["evaluation"]
models = config["models"]
manifest_path = ROOT / "results/model-manifest-5080-20260924.json"
manifest = json.loads(manifest_path.read_text())
prompt_path = ROOT / evaluation["prompt_manifest"]
prompts = load_prompts(prompt_path)
prompt = prompts[0]
torch.manual_seed(evaluation["seed"])
model = load_official_eagle3(
    ROOT / models["target_dir"],
    ROOT / models["draft_dir"],
    angelslim_revision=models["angelslim_revision"],
    total_token=evaluation["total_token"],
    depth=evaluation["depth"],
    top_k=evaluation["top_k"],
    threshold=evaluation["threshold"],
    target_load_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
)
model.eval()
device = next(model.base_model.parameters()).device
input_ids = encode_prompt(model.tokenizer, prompt["messages"], evaluation["thinking_mode"], device)

tree_contexts = []
posterior_rows = []
original_tree_decoding = eagle3_mod.tree_decoding
original_evaluate_posterior = eagle3_mod.evaluate_posterior

def capture_tree_decoding(*args, **kwargs):
    tree_position_ids = args[3].detach().cpu().reshape(-1).tolist()
    input_context = args[4]
    retrieve_indices = args[5].detach().cpu().tolist()
    result = original_tree_decoding(*args, **kwargs)
    tree_contexts.append({
        "context_length_before_tree": int(input_context.shape[1]),
        "tree_position_ids": [int(x) for x in tree_position_ids],
        "retrieve_indices": [[int(x) for x in row] for row in retrieve_indices],
    })
    return result

def capture_posterior(logits, candidates, logits_processor):
    result = original_evaluate_posterior(logits, candidates, logits_processor)
    best, accepted, sample = result
    best_index = int(best.item()) if torch.is_tensor(best) else int(best)
    accepted_count = int(accepted.item()) if torch.is_tensor(accepted) else int(accepted)
    context = tree_contexts[-1]
    retrieve_path = context["retrieve_indices"][best_index]
    verifier_slot = retrieve_path[accepted_count]
    verifier_position_id = (
        context["context_length_before_tree"] + context["tree_position_ids"][verifier_slot]
    )
    verifier_logits = logits[best_index, accepted_count].detach().float()
    top_values, top_ids = torch.topk(verifier_logits, 10)
    sample_id = int(sample.reshape(-1)[0].item())
    posterior_rows.append({
        "round": len(posterior_rows),
        "context_length_before_tree": context["context_length_before_tree"],
        "accepted_draft_count": accepted_count,
        "best_candidate_index": best_index,
        "accepted_candidate_path_ids": [
            int(x) for x in candidates[best_index, : accepted_count + 1].detach().cpu().tolist()
        ],
        "verifier_slot": int(verifier_slot),
        "tree_position_id": int(context["tree_position_ids"][verifier_slot]),
        "absolute_position_id": int(verifier_position_id),
        "target_selected_seed_id": sample_id,
        "verifier_logit_dtype": str(logits.dtype),
        "verifier_top10": [
            {"token_id": int(token_id), "logit": float(value)}
            for value, token_id in zip(top_values.cpu().tolist(), top_ids.cpu().tolist())
        ],
        "verifier_logits_of_target_272_and_eagle_11": {
            "target_272": float(verifier_logits[272].item()),
            "eagle_11": float(verifier_logits[11].item()),
        },
    })
    return result

eagle3_mod.tree_decoding = capture_tree_decoding
eagle3_mod.evaluate_posterior = capture_posterior
with torch.inference_mode():
    output, new_tokens, step, accept_lengths = model.eagle_generate(
        input_ids,
        temperature=0.0,
        max_new_tokens=3,
        max_length=evaluation["max_length"],
        log=True,
    )
torch.cuda.synchronize()
generated = [int(x) for x in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
prefix = output[:, : input_ids.shape[1] + 3]
reset_tree_mode(model)
with torch.inference_mode():
    full_prefix_output = model.base_model(prefix, use_cache=False)
torch.cuda.synchronize()
full_logits = full_prefix_output.logits[0, -1].detach().float()
full_values, full_ids = torch.topk(full_logits, 10)
trace = {
    "created_at_utc": datetime.now(UTC).isoformat(),
    "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "target_revision": models["target_revision"],
    "draft_revision": models["draft_revision"],
    "angelslim_revision": models["angelslim_revision"],
    "model_manifest_sha256": sha256_file(manifest_path),
    "config_sha256": sha256_file(config_path),
    "prompt_sha256": sha256_file(prompt_path),
    "prompt_id": prompt["id"],
    "precision": {
        "target_dtype": str(model.base_model.dtype),
        "draft_dtype": str(next(model.eagle_layer.parameters()).dtype),
        "verifier_logits_dtype": posterior_rows[0]["verifier_logit_dtype"] if posterior_rows else None,
        "full_prefix_logits_dtype": str(full_prefix_output.logits.dtype),
    },
    "device": torch.cuda.get_device_name(0),
    "diagnostic_settings": {
        "temperature": 0.0,
        "max_new_tokens": 3,
        "total_token": evaluation["total_token"],
        "depth": evaluation["depth"],
        "top_k": evaluation["top_k"],
        "threshold": evaluation["threshold"],
    },
    "prompt_input_token_count": int(input_ids.shape[1]),
    "generated_token_ids": generated,
    "accepted_draft_counts_per_round": [
        int(x.item()) if torch.is_tensor(x) else int(x) for x in accept_lengths
    ],
    "captured_posterior_rounds": posterior_rows,
    "full_prefix_token_count": int(prefix.shape[1]),
    "full_prefix_generated_token_ids": [
        int(x) for x in prefix[0, input_ids.shape[1]:].detach().cpu().tolist()
    ],
    "full_prefix_target_top10": [
        {"token_id": int(token_id), "logit": float(value)}
        for value, token_id in zip(full_values.cpu().tolist(), full_ids.cpu().tolist())
    ],
    "full_prefix_target_logits_of_272_and_11": {
        "target_272": float(full_logits[272].item()),
        "eagle_11": float(full_logits[11].item()),
    },
}
out = ROOT / "results/cuda-parity-diagnostic-20260924/trace.json"
out.write_text(json.dumps(trace, indent=2) + "\n")
print(json.dumps({
    "trace": str(out),
    "first_round_accept_lengths": trace["accepted_draft_counts_per_round"],
    "first_round_target_seed": posterior_rows[0]["target_selected_seed_id"] if posterior_rows else None,
    "first_round_verifier_logits": posterior_rows[0]["verifier_logits_of_target_272_and_eagle_11"] if posterior_rows else None,
    "full_prefix_logits": trace["full_prefix_target_logits_of_272_and_11"],
}, indent=2))
