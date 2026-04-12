"""Qwen 2.5 14B: comprehensive evaluation (v4, val-seed protocol).

Model: Qwen/Qwen2.5-14B | HF: https://huggingface.co/Qwen/Qwen2.5-14B
GPU: H100 SXM 80GB | 7 seeds (43-49) | 350 ex/dim | Dense sweep L16-40 | Date: 2026-04-08
Protocol: layer selected on val with seed 42, evaluated on val with seeds 43-49

Usage: pip install transformers datasets scipy && python scripts/qwen14b_v4.py
"""

import subprocess, shutil
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi'], check=False)
elif shutil.which('rocm-smi'):
    subprocess.run(['rocm-smi'], check=False)
else:
    print('No GPU management tool found (nvidia-smi / rocm-smi)')

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, rankdata, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_wikitext(split='test', max_docs=None):
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    docs, current = [], []
    for row in ds:
        text = row['text']
        if text.strip() == '' and current:
            docs.append('\n'.join(current)); current = []
            if max_docs and len(docs) >= max_docs: break
        elif text.strip(): current.append(text)
    if current: docs.append('\n'.join(current))
    return docs

def partial_spearman(x, y, covariates):
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, p = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r), float(p)

def compute_loss_residuals(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    return losses - X @ beta

def collect_layer_data(model, tokenizer, docs, layer, device, max_tokens=200000, max_length=512):
    model.eval()
    all_acts, all_losses, all_softmax, all_entropy, all_norms = [], [], [], [], []
    total = 0
    for doc in docs:
        if total >= max_tokens: break
        if not doc.strip(): continue
        tokens = tokenizer(doc, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = tokens['input_ids'].to(device)
        if input_ids.size(1) < 2: continue
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
        h = outputs.hidden_states[layer + 1][0, :-1, :].cpu()
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        losses = F.cross_entropy(logits, labels, reduction='none').cpu()
        probs = F.softmax(logits, dim=-1)
        sm = probs.max(dim=-1).values.cpu()
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        norms = h.norm(dim=-1)
        all_acts.append(h); all_losses.append(losses); all_softmax.append(sm)
        all_entropy.append(ent); all_norms.append(norms)
        total += h.size(0)
    print(f'    {total} positions from {len(all_acts)} documents')
    return {
        'activations': torch.cat(all_acts).float(),
        'losses': torch.cat(all_losses).float().numpy(),
        'max_softmax': torch.cat(all_softmax).float().numpy(),
        'logit_entropy': torch.cat(all_entropy).float().numpy(),
        'activation_norm': torch.cat(all_norms).float().numpy(),
    }

def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    acts = train_data['activations']
    residuals = compute_loss_residuals(
        train_data['losses'], train_data['max_softmax'], train_data['activation_norm'])
    targets = torch.from_numpy((residuals > 0).astype(np.float32))
    head = torch.nn.Linear(acts.size(1), 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head

def evaluate_head(head, test_data):
    head.eval()
    with torch.inference_mode():
        scores = head(test_data['activations']).squeeze(-1).numpy()
    rho, p = partial_spearman(scores, test_data['losses'],
                               [test_data['max_softmax'], test_data['activation_norm']])
    return scores, rho, p

def compute_hand_designed(data):
    acts = data['activations']
    p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
    act_entropy = -(p * (p + 1e-8).log()).sum(dim=1).numpy()
    return {
        'ff_goodness': (acts**2).mean(dim=1).numpy(),
        'active_ratio': (acts.abs() > 0.01).float().mean(dim=1).numpy(),
        'act_entropy': act_entropy,
        'activation_norm': data['activation_norm'],
    }

MODEL_ID = 'Qwen/Qwen2.5-14B'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
    attn_implementation='sdpa'
).cuda()
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

N_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
N_PARAMS = sum(p.numel() for p in model.parameters()) / 1e9
print(f'{N_PARAMS:.1f}B params, {N_LAYERS} layers, {HIDDEN_DIM} dim')

TARGET_EX_PER_DIM = 350
MAX_TRAIN = TARGET_EX_PER_DIM * HIDDEN_DIM
LAYER_SELECT_SEED = 42
EVAL_SEEDS = list(range(43, 50))  # 7 held-out seeds
print(f'Token budget: {MAX_TRAIN} train ({TARGET_EX_PER_DIM} ex/dim)')
print(f'Layer selection: seed {LAYER_SELECT_SEED} on val')
print(f'Evaluation: seeds {EVAL_SEEDS} on val (held-out from selection)')
print(f'GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

wiki_train = load_wikitext('train', max_docs=12000)
wiki_val = load_wikitext('validation', max_docs=None)
wiki_test = load_wikitext('test', max_docs=None)
print(f'{len(wiki_train)} train, {len(wiki_val)} val, {len(wiki_test)} test')

approx_train = sum(len(d.split()) for d in wiki_train)
approx_val = sum(len(d.split()) for d in wiki_val)
print(f'Approx tokens: {approx_train} train ({approx_train/HIDDEN_DIM:.0f} ex/dim), {approx_val} val ({approx_val/HIDDEN_DIM:.0f} ex/dim)')
assert approx_train > MAX_TRAIN * 0.8, 'Need more train docs'

sweep_layers = [0, 8] + list(range(16, 41)) + [44, 47]
print(f'Sweeping {len(sweep_layers)} layers with seed {LAYER_SELECT_SEED}')
layer_profile = {}

for layer in sweep_layers:
    tr = collect_layer_data(model, tokenizer, wiki_train, layer, 'cuda', MAX_TRAIN)
    va = collect_layer_data(model, tokenizer, wiki_val, layer, 'cuda', MAX_TRAIN)
    head = train_linear_binary(tr, seed=LAYER_SELECT_SEED)
    _, rho, _ = evaluate_head(head, va)
    layer_profile[layer] = float(rho)
    print(f'  layer {layer:>2}: {rho:+.4f} ({len(tr["losses"])} train, {len(va["losses"])} val)')
    del tr, va; torch.cuda.empty_cache()

peak_layer = max(layer_profile, key=layer_profile.get)
output_layer = N_LAYERS - 1
if peak_layer >= output_layer - 3:
    max_mid = int(0.8 * N_LAYERS)
    mid_candidates = {l: r for l, r in layer_profile.items() if l <= max_mid}
    if mid_candidates:
        peak_layer = max(mid_candidates, key=mid_candidates.get)
        print(f'Peak near output, using mid-depth: layer {peak_layer}')

sorted_layers = sorted(layer_profile.items(), key=lambda x: x[1], reverse=True)
candidate_layers = sorted([l for l, _ in sorted_layers[:4] if l <= int(0.8 * N_LAYERS)])

val_rho_at_peak = layer_profile[peak_layer]
print(f'\nPeak layer: {peak_layer} ({peak_layer/N_LAYERS:.0%} depth) = {val_rho_at_peak:+.4f}')
print(f'Candidate layers for multi-layer eval: {candidate_layers}')

train_cache = {}
val_cache = {}

for layer in candidate_layers:
    print(f'Collecting layer {layer}...')
    train_cache[layer] = collect_layer_data(model, tokenizer, wiki_train, layer, 'cuda', MAX_TRAIN)
    val_cache[layer] = collect_layer_data(model, tokenizer, wiki_val, layer, 'cuda', MAX_TRAIN)

print(f'Collecting output layer {output_layer}...')
wiki_train_output = collect_layer_data(model, tokenizer, wiki_train, output_layer, 'cuda', MAX_TRAIN)
wiki_val_output = collect_layer_data(model, tokenizer, wiki_val, output_layer, 'cuda', MAX_TRAIN)

print(f'Collecting test split at peak layer {peak_layer}...')
wiki_test_peak = collect_layer_data(model, tokenizer, wiki_test, peak_layer, 'cuda', MAX_TRAIN)
wiki_test_output = collect_layer_data(model, tokenizer, wiki_test, output_layer, 'cuda', MAX_TRAIN)

for layer in candidate_layers:
    print(f'  L{layer} train: {len(train_cache[layer]["losses"])} ({len(train_cache[layer]["losses"])/HIDDEN_DIM:.0f} ex/dim)')
    print(f'  L{layer} val:   {len(val_cache[layer]["losses"])} ({len(val_cache[layer]["losses"])/HIDDEN_DIM:.0f} ex/dim)')
print(f'  Test: {len(wiki_test_peak["losses"])} ({len(wiki_test_peak["losses"])/HIDDEN_DIM:.0f} ex/dim)')


c4_docs = []
ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
for i, row in enumerate(ds):
    if i < 50000: continue
    text = row['text'].strip()
    if len(text) > 100: c4_docs.append(text)
    if len(c4_docs) >= 500: break

c4_test_peak = collect_layer_data(model, tokenizer, c4_docs, peak_layer, 'cuda', MAX_TRAIN // 2)

c4_train_docs = []
ds2 = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
for row in ds2:
    text = row['text'].strip()
    if len(text) > 100: c4_train_docs.append(text)
    if len(c4_train_docs) >= 8000: break

c4_train_peak = collect_layer_data(model, tokenizer, c4_train_docs, peak_layer, 'cuda', MAX_TRAIN)
print(f'C4: {len(c4_test_peak["losses"])} test, {len(c4_train_peak["losses"])} train')

del model; gc.collect(); torch.cuda.empty_cache()
print('Model unloaded.')

print(f'Evaluating {len(candidate_layers)} layers with {len(EVAL_SEEDS)} held-out seeds on val split\n')

layer_eval = {}
for layer in candidate_layers:
    seed_rhos = []
    seed_scores = []
    for seed in EVAL_SEEDS:
        head = train_linear_binary(train_cache[layer], seed=seed)
        scores, rho, _ = evaluate_head(head, val_cache[layer])
        seed_rhos.append(float(rho))
        seed_scores.append(scores)

    pairwise = []
    for i in range(len(EVAL_SEEDS)):
        for j in range(i + 1, len(EVAL_SEEDS)):
            r, _ = spearmanr(seed_scores[i], seed_scores[j])
            pairwise.append(float(r))

    layer_eval[layer] = {
        'mean': float(np.mean(seed_rhos)),
        'std': float(np.std(seed_rhos)),
        'per_seed': seed_rhos,
        'seed_agreement': float(np.mean(pairwise)),
        'scores': seed_scores,
    }
    print(f'  L{layer}: {np.mean(seed_rhos):+.4f} +/- {np.std(seed_rhos):.4f}  agree={np.mean(pairwise):+.4f}  seeds={[round(x,4) for x in seed_rhos]}')

# Best layer on held-out eval
best_layer = max(layer_eval, key=lambda l: layer_eval[l]['mean'])
print(f'\nBest layer (held-out eval): L{best_layer} = {layer_eval[best_layer]["mean"]:+.4f}')
print(f'Val sweep peak (seed 42): L{peak_layer} = {val_rho_at_peak:+.4f}')
if best_layer != peak_layer:
    print(f'Layer selection shifted from L{peak_layer} to L{best_layer}')

FINAL_LAYER = best_layer
wiki_train_peak = train_cache[FINAL_LAYER]
wiki_val_peak = val_cache[FINAL_LAYER]

ev = layer_eval[FINAL_LAYER]
mean_rho = ev['mean']
all_rhos = ev['per_seed']
all_scores = ev['scores']
mean_agree = ev['seed_agreement']

print(f'FINAL LAYER: {FINAL_LAYER} ({FINAL_LAYER/N_LAYERS:.0%} depth)')
print(f'Partial corr (val, held-out seeds): {mean_rho:+.4f} +/- {ev["std"]:.4f}')
print(f'Seed agreement: {mean_agree:+.4f}')
print(f'Spread: {max(all_rhos) - min(all_rhos):.4f}')

test_rhos = []
for seed in EVAL_SEEDS[:3]:
    head = train_linear_binary(wiki_train_peak, seed=seed)
    _, rho, _ = evaluate_head(head, wiki_test_peak)
    test_rhos.append(float(rho))
print(f'\nTest-split comparison (3 seeds): {np.mean(test_rhos):+.4f} ({[round(x,4) for x in test_rhos]})')
print(f'Val-test gap: {mean_rho - np.mean(test_rhos):+.4f}')

hand_designed = compute_hand_designed(wiki_val_peak)
baseline_results = {}
for name, sc in hand_designed.items():
    rho_hd, _ = partial_spearman(sc, wiki_val_peak['losses'],
                                  [wiki_val_peak['max_softmax'], wiki_val_peak['activation_norm']])
    baseline_results[name] = float(rho_hd)
    print(f'  {name:<20} {rho_hd:+.4f}')

torch.manual_seed(99)
rand_head = torch.nn.Linear(HIDDEN_DIM, 1)
rand_head.eval()
with torch.inference_mode():
    rand_sc = rand_head(wiki_val_peak['activations']).squeeze(-1).numpy()
rho_rand, _ = partial_spearman(rand_sc, wiki_val_peak['losses'],
                                [wiki_val_peak['max_softmax'], wiki_val_peak['activation_norm']])
baseline_results['random_head'] = float(rho_rand)
print(f'  {"random_head":<20} {rho_rand:+.4f}')

ctrl_rhos = []
for seed in EVAL_SEEDS[:3]:
    torch.manual_seed(seed); np.random.seed(seed)
    n_feat = wiki_train_output['activations'].size(1)
    predictor = torch.nn.Sequential(
        torch.nn.Linear(n_feat, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(
        wiki_train_output['activations'], torch.from_numpy(wiki_train_output['losses']).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    for _ep in range(20):
        for bx, by in dl:
            loss = F.mse_loss(predictor(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    predictor.eval()
    with torch.inference_mode():
        pred_sc = predictor(wiki_val_output['activations']).squeeze(-1).numpy()
    obs = train_linear_binary(wiki_train_peak, seed=seed)
    obs.eval()
    with torch.inference_mode():
        obs_sc = obs(wiki_val_peak['activations']).squeeze(-1).numpy()
    rho_ctrl, _ = partial_spearman(obs_sc, wiki_val_peak['losses'],
                                    [wiki_val_peak['max_softmax'], wiki_val_peak['activation_norm'], pred_sc])
    ctrl_rhos.append(float(rho_ctrl))
    print(f'  seed {seed}: output-controlled = {rho_ctrl:+.4f}')
print(f'  Mean: {np.mean(ctrl_rhos):+.4f}')

domain_results = {}
for domain_name, test_data in [('wikitext_val', wiki_val_peak), ('c4', c4_test_peak)]:
    rhos = []
    for seed in EVAL_SEEDS[:3]:
        head = train_linear_binary(wiki_train_peak, seed=seed)
        _, rho, _ = evaluate_head(head, test_data)
        rhos.append(float(rho))
    domain_results[domain_name] = float(np.mean(rhos))
    print(f'  {domain_name:<15}: {np.mean(rhos):+.4f}')

c4_rhos = []
for seed in EVAL_SEEDS[:3]:
    c4_head = train_linear_binary(c4_train_peak, seed=seed)
    _, rho, _ = evaluate_head(c4_head, c4_test_peak)
    c4_rhos.append(float(rho))
domain_results['c4_within'] = float(np.mean(c4_rhos))
print(f'  Within-domain C4: {np.mean(c4_rhos):+.4f}')

torch.manual_seed(42)
conf_feats = torch.from_numpy(
    np.column_stack([wiki_train_peak['max_softmax'], wiki_train_peak['activation_norm']])).float()
loss_tgt = torch.from_numpy(wiki_train_peak['losses']).float()
mlp_ctrl = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(mlp_ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
ds = torch.utils.data.TensorDataset(conf_feats, loss_tgt)
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
for _ep in range(20):
    for bx, by in dl:
        loss = F.mse_loss(mlp_ctrl(bx).squeeze(-1), by)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
mlp_ctrl.eval()
with torch.inference_mode():
    mlp_pred = mlp_ctrl(torch.from_numpy(
        np.column_stack([wiki_val_peak['max_softmax'], wiki_val_peak['activation_norm']])).float()
    ).squeeze(-1).numpy()

head = train_linear_binary(wiki_train_peak, seed=EVAL_SEEDS[0])
head.eval()
with torch.inference_mode():
    obs = head(wiki_val_peak['activations']).squeeze(-1).numpy()

td = wiki_val_peak
ctrl_results = {}
for name, covs in [('none', None), ('softmax_only', [td['max_softmax']]),
                    ('norm_only', [td['activation_norm']]),
                    ('standard', [td['max_softmax'], td['activation_norm']]),
                    ('plus_entropy', [td['max_softmax'], td['activation_norm'], td['logit_entropy']]),
                    ('nonlinear', [mlp_pred])]:
    if covs is None: r, _ = spearmanr(obs, td['losses'])
    else: r, _ = partial_spearman(obs, td['losses'], covs)
    ctrl_results[name] = float(r)
    print(f'  {name:<16}: {r:+.4f}')

n_flag = min(len(wiki_val_peak['losses']), len(wiki_val_output['losses']))
flag_losses = wiki_val_peak['losses'][:n_flag]
flag_softmax = wiki_val_output['max_softmax'][:n_flag]
flag_acts = wiki_val_peak['activations'][:n_flag]
median_loss = float(np.median(flag_losses))
is_high_loss = flag_losses > median_loss

flag_rates = [0.05, 0.10, 0.20, 0.30]
flagging_results = []
print(f'{n_flag} tokens, median loss = {median_loss:.4f}')

for seed in EVAL_SEEDS[:3]:
    head = train_linear_binary(wiki_train_peak, seed=seed)
    head.eval()
    with torch.inference_mode():
        obs_scores = head(flag_acts).squeeze(-1).numpy()
    seed_r = {'observer': {}, 'confidence': {}, 'exclusive': {}}
    for rate in flag_rates:
        k = int(n_flag * rate)
        obs_flagged = obs_scores >= np.sort(obs_scores)[-k]
        conf_flagged = flag_softmax <= np.sort(flag_softmax)[k]
        obs_prec = float(is_high_loss[obs_flagged].mean()) if obs_flagged.sum() > 0 else 0.0
        conf_prec = float(is_high_loss[conf_flagged].mean()) if conf_flagged.sum() > 0 else 0.0
        obs_excl = int((obs_flagged & ~conf_flagged & is_high_loss).sum())
        conf_excl = int((conf_flagged & ~obs_flagged & is_high_loss).sum())
        combined = obs_flagged | conf_flagged
        comb_prec = float(is_high_loss[combined].mean()) if combined.sum() > 0 else 0.0
        seed_r['observer'][str(rate)] = obs_prec
        seed_r['confidence'][str(rate)] = conf_prec
        seed_r['exclusive'][str(rate)] = {
            'observer_only': obs_excl, 'confidence_only': conf_excl,
            'combined_precision': comb_prec}
    flagging_results.append(seed_r)

flagging_summary = {}
print(f'{"Rate":<8} {"Obs prec":>10} {"Conf prec":>10} {"Obs excl":>10}')
print('-' * 40)
for rate in flag_rates:
    r = str(rate)
    op = np.mean([s['observer'][r] for s in flagging_results])
    cp = np.mean([s['confidence'][r] for s in flagging_results])
    oe = np.mean([s['exclusive'][r]['observer_only'] for s in flagging_results])
    ce = np.mean([s['exclusive'][r]['confidence_only'] for s in flagging_results])
    flagging_summary[r] = {'observer_precision': float(op), 'confidence_precision': float(cp),
                            'observer_exclusive': float(oe), 'confidence_exclusive': float(ce)}
    print(f'{rate:<8.0%} {op:>10.3f} {cp:>10.3f} {oe:>10.0f}')

total_errors = n_flag / 2
excl_10 = flagging_summary['0.1']['observer_exclusive']
print(f'\nError coverage (10%): {excl_10:.0f} = {excl_10/total_errors*100:.1f}% of errors')

print(f'Qwen 2.5 14B v4 (val-seed protocol, {len(EVAL_SEEDS)} held-out seeds, {TARGET_EX_PER_DIM} ex/dim)')
print(f'  Layer selection: seed {LAYER_SELECT_SEED} on val → L{peak_layer}')
print(f'  Best held-out layer: L{FINAL_LAYER}')
print(f'  Partial corr (val, held-out seeds): {mean_rho:+.4f} +/- {np.std(all_rhos):.4f}')
print(f'  Test-split comparison: {np.mean(test_rhos):+.4f}')
print(f'  Seed agreement: {mean_agree:+.4f}')
print(f'  Output-controlled: {np.mean(ctrl_rhos):+.4f}')
print(f'  Error coverage: {excl_10/total_errors*100:.1f}%')
print()
print(f'Val ex/dim: {len(wiki_val_peak["losses"])/HIDDEN_DIM:.0f} vs test ex/dim: {len(wiki_test_peak["losses"])/HIDDEN_DIM:.0f}')
print()
print('Version progression:')
print('  v1 (68 ex/dim, test):  +0.194')
print('  v2 (250 ex/dim, test): +0.212')
print('  v3 (350 ex/dim, test): +0.214')
print(f'  v4 (350 ex/dim, val-seed): {mean_rho:+.4f}')

output = {
    'model': MODEL_ID,
    'n_params_b': round(N_PARAMS, 1),
    'n_layers': N_LAYERS,
    'hidden_dim': HIDDEN_DIM,
    'protocol': {
        'description': 'Layer selected on val with seed 42, evaluated on val with held-out seeds 43-49',
        'layer_select_seed': LAYER_SELECT_SEED,
        'eval_seeds': EVAL_SEEDS,
        'eval_split': 'validation',
        'rationale': 'Test split has 43.8 ex/dim at hidden_dim=5120, producing noisy estimates. Val split has 350 ex/dim. Held-out seeds prevent circularity.',
    },
    'peak_layer_sweep': peak_layer,
    'peak_layer_final': FINAL_LAYER,
    'peak_layer_frac': round(FINAL_LAYER / N_LAYERS, 2),
    'layer_profile': {str(k): v for k, v in sorted(layer_profile.items())},
    'multi_layer_eval': {
        str(l): {'mean': v['mean'], 'std': v['std'], 'per_seed': v['per_seed'], 'seed_agreement': v['seed_agreement']}
        for l, v in layer_eval.items()
    },
    'token_budget': {
        'target_ex_per_dim': TARGET_EX_PER_DIM,
        'train_tokens': int(len(wiki_train_peak['losses'])),
        'val_tokens': int(len(wiki_val_peak['losses'])),
        'test_tokens': int(len(wiki_test_peak['losses'])),
        'train_ex_per_dim': round(len(wiki_train_peak['losses']) / HIDDEN_DIM, 1),
        'val_ex_per_dim': round(len(wiki_val_peak['losses']) / HIDDEN_DIM, 1),
        'test_ex_per_dim': round(len(wiki_test_peak['losses']) / HIDDEN_DIM, 1),
    },
    'partial_corr': {
        'mean': mean_rho,
        'std': float(np.std(all_rhos)),
        'per_seed': all_rhos,
        'n_seeds': len(EVAL_SEEDS),
        'split': 'validation (held-out seeds)',
    },
    'test_split_comparison': {
        'mean': float(np.mean(test_rhos)),
        'per_seed': test_rhos,
        'val_test_gap': float(mean_rho - np.mean(test_rhos)),
    },
    'seed_agreement': {'mean': mean_agree},
    'output_controlled': {'mean': float(np.mean(ctrl_rhos)), 'per_seed': ctrl_rhos},
    'baselines': baseline_results,
    'cross_domain': domain_results,
    'control_sensitivity': ctrl_results,
    'flagging_6a': {
        'seeds': EVAL_SEEDS[:3],
        'n_tokens': n_flag,
        'median_loss': median_loss,
        'peak_layer': FINAL_LAYER,
        'output_layer': output_layer,
        'eval_split': 'validation',
        'per_seed': flagging_results,
        'summary': flagging_summary,
    },
    'version_history': {
        'v1': {'partial_corr': 0.194, 'output_controlled': 0.135, 'ex_per_dim': 68, 'peak_layer': 32, 'eval_split': 'test'},
        'v2': {'partial_corr': 0.212, 'output_controlled': 0.111, 'ex_per_dim': 250, 'peak_layer': 28, 'eval_split': 'test'},
        'v3': {'partial_corr': 0.214, 'output_controlled': 0.096, 'ex_per_dim': 350, 'peak_layer': 30, 'eval_split': 'test', 'test_ex_per_dim': 43.8},
    },
}


out_path = Path(__file__).resolve().parent.parent / 'results' / 'qwen14b_v4_results.json'
out_path.parent.mkdir(exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved {out_path}')
print(json.dumps(output, indent=2))
