import os
import sys
import json
import numpy as np
from docopt import docopt

# Make models/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from EnsembleModel import (
    soft_voting,
    weighted_voting,
    StackingMeta,
    evaluate_predictions,
    compute_diversity_matrix,
)


# Model taxonomy


ALL_MODELS = ["RNN", "LSTM", "FCN", "DKVMN", "SAKT", "DKTPlus", "KQN"]

MODEL_FAMILIES = {
    "RNN_based":   ["RNN", "LSTM", "DKTPlus"],   # DKTPlus uses an RNN backbone
    "Attention":   ["SAKT"],
    "Memory":      ["DKVMN"],
    "Convolutional": ["FCN"],
    "Latent": ["KQN"]
}

# Pre-defined groups (top3 is built dynamically per dataset)
STATIC_GROUPS = {
    "all_models":   ALL_MODELS,
    "rnn_family":   MODEL_FAMILIES["RNN_based"],
    "cross_family": ["LSTM", "SAKT", "DKVMN"],        # one per different family
    "diverse":      ["LSTM", "SAKT", "DKVMN", "FCN", "KQN"], # all non-RNN families + best RNN
}


def load_preds(preds_dir: str, dataset: str, model: str, run: int, split: str):
    p_path = os.path.join(preds_dir, dataset, f"{model}_run{run}_{split}_preds.npy")
    i_path = os.path.join(preds_dir, dataset, f"{model}_run{run}_{split}_ids.npy")
    l_path = os.path.join(preds_dir, dataset, f"{model}_run{run}_{split}_labels.npy")
    if not all(os.path.exists(path) for path in (p_path, i_path, l_path)):
        return None, None, None
    return np.load(p_path), np.load(i_path), np.load(l_path)


def load_individual_auc(results_dir: str, dataset: str, model: str) -> float:
    path = os.path.join(results_dir, "individual", dataset, f"{model}.json")
    if not os.path.exists(path):
        return 0.5                # neutral weight if missing
    with open(path) as f:
        return json.load(f).get("avg_auc", 0.5)


def save_json(data: dict, *path_parts):
    path = os.path.join(*path_parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def aggregate_predictions(preds_dir, dataset, models, n_runs, split):
    prediction_maps, label_maps, found = [], [], []

    for model in models:
        run_preds, run_ids, run_labels = [], [], []
        for run in range(n_runs):
            p, ids, labels = load_preds(preds_dir, dataset, model, run, split)
            if p is None:
                break
            run_preds.append(p)
            run_ids.append(ids)
            run_labels.append(labels)

        if len(run_preds) != n_runs:
            print(f"    [warn] no complete ID-indexed {split} predictions for "
                  f"{model} on {dataset} — skipped")
            continue

        if any(not np.array_equal(ids, run_ids[0]) for ids in run_ids[1:]):
            print(f"    [skip] interaction IDs differ between runs for {model} — skipped")
            continue
        if any(not np.array_equal(labels, run_labels[0]) for labels in run_labels[1:]):
            print(f"    [skip] labels differ between runs for {model} — skipped")
            continue

        valid = run_ids[0] >= 0
        averaged = np.mean(run_preds, axis=0)
        if len(averaged) != len(run_ids[0]) or len(averaged) != len(run_labels[0]):
            print(f"    [skip] prediction, label, and ID lengths differ for {model} — skipped")
            continue
        prediction_maps.append(dict(zip(run_ids[0][valid], averaged[valid])))
        label_maps.append(dict(zip(run_ids[0][valid], run_labels[0][valid])))
        found.append(model)

    if not prediction_maps:
        return [], None, []

    common_ids = set(prediction_maps[0])
    for prediction_map in prediction_maps[1:]:
        common_ids.intersection_update(prediction_map)
    common_ids = np.array(sorted(common_ids), dtype=np.int64)

    if not len(common_ids):
        print(f"    [skip] no shared valid {split} interactions across models")
        return [], None, []

    labels_ref = np.array([label_maps[0][identifier] for identifier in common_ids])
    for model, label_map in zip(found[1:], label_maps[1:]):
        labels = np.array([label_map[identifier] for identifier in common_ids])
        if not np.array_equal(labels, labels_ref):
            print(f"    [skip] inconsistent labels for {model} on shared interactions")
            return [], None, []

    preds_list = [np.array([prediction_map[identifier] for identifier in common_ids])
                  for prediction_map in prediction_maps]
    print(f"    [info] aligned {len(common_ids)} shared valid {split} interactions")
    return preds_list, labels_ref, found



# Build model groups for a given dataset


def build_groups(results_dir, dataset, available_models):
    individual_aucs = {
        m: load_individual_auc(results_dir, dataset, m)
        for m in available_models
    }

    # Dynamic: top-3 by individual AUC
    sorted_models = sorted(individual_aucs.items(), key=lambda x: x[1], reverse=True)
    top3 = [m for m, _ in sorted_models[:3] if m in available_models]

    groups = {**STATIC_GROUPS, "top3": top3}

    # Filter to available models only and require >= 2 members
    valid = {}
    for name, members in groups.items():
        present = [m for m in members if m in available_models]
        if len(present) >= 2:
            valid[name] = present
        else:
            print(f"    [skip] group '{name}' has fewer than 2 available models")

    return valid, individual_aucs



# Run all strategies for one group


def run_group(
    group_name, models, dataset,
    preds_dir, results_dir, n_runs,
    individual_aucs, strategies,
):
    print(f"\n    Group [{group_name}] → {models}")

    # Load test predictions (averaged over runs)
    test_preds, test_labels, test_found = aggregate_predictions(
        preds_dir, dataset, models, n_runs, "test"
    )
    if len(test_found) < 2:
        print(f"      Not enough models with test predictions — skip")
        return None

    # Load val predictions (for stacking)
    val_preds, val_labels, val_found = aggregate_predictions(
        preds_dir, dataset, test_found, n_runs, "val"
    )
    # Align val_found with test_found (keep same order)
    if val_found != test_found:
        print(f"      [warn] val/test model lists differ — stacking may be skipped")

    # Weights from individual AUC
    weights = [individual_aucs.get(m, 0.5) for m in test_found]

    group_result = {
        "dataset":  dataset,
        "group":    group_name,
        "models":   test_found,
        "n_models": len(test_found),
    }

    #1. Soft Voting
    if "soft" in strategies or "all" in strategies:
        sv_preds = soft_voting(test_preds)
        auc, f1, acc = evaluate_predictions(sv_preds, test_labels)
        group_result["soft_voting"] = {"auc": auc, "f1": f1, "accuracy": acc}
        print(f"      soft_voting    → AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

    #2. Weighted Voting
    if "weighted" in strategies or "all" in strategies:
        wv_preds = weighted_voting(test_preds, weights)
        auc, f1, acc = evaluate_predictions(wv_preds, test_labels)
        group_result["weighted_voting"] = {
            "auc": auc, "f1": f1, "accuracy": acc,
            "weights": {m: round(w, 4) for m, w in zip(test_found, weights)},
        }
        print(f"      weighted_voting → AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

    #3. Stacking
    if "stacking" in strategies or "all" in strategies:
        if len(val_found) >= 2 and val_labels is not None:
            # Keep only models present in both val and test
            common = [m for m in test_found if m in val_found]
            vp_aligned  = [val_preds[val_found.index(m)]  for m in common]
            tp_aligned  = [test_preds[test_found.index(m)] for m in common]

            if len(common) >= 2:
                stacker = StackingMeta()
                stacker.fit(vp_aligned, val_labels)
                st_preds = stacker.predict(tp_aligned)
                auc, f1, acc = evaluate_predictions(st_preds, test_labels)
                group_result["stacking"] = {
                    "auc": auc, "f1": f1, "accuracy": acc,
                    "meta_weights": {
                        m: round(float(c), 4)
                        for m, c in zip(common, stacker.meta.coef_[0])
                    },
                }
                print(f"      stacking       → AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
            else:
                print(f"      stacking skipped — not enough models in both val and test")
        else:
            print(f"      stacking skipped — val predictions not found")

    # ── Diversity matrix──
    group_result["diversity"] = compute_diversity_matrix(test_preds, test_found)

    # Save per-strategy files
    out_base = os.path.join(results_dir, "ensemble")
    for strat in ["soft_voting", "weighted_voting", "stacking"]:
        if strat not in group_result:
            continue
        save_json(
            {**group_result["soft_voting" if strat == "soft_voting" else strat],
             "dataset": dataset, "group": group_name, "models": test_found},
            out_base, strat, dataset, f"{group_name}.json",
        )

    return group_result



# Summary builders (RQ1, RQ2, RQ3)


def build_rq1_summary(all_results, results_dir):
    summary = {}
    for dataset, groups in all_results.items():
        best_ind_auc = 0.0
        # Individual models
        ind_dir = os.path.join(results_dir, "individual", dataset)
        if os.path.isdir(ind_dir):
            for fname in os.listdir(ind_dir):
                with open(os.path.join(ind_dir, fname)) as f:
                    d = json.load(f)
                best_ind_auc = max(best_ind_auc, d.get("avg_auc", 0))

        best_ens_auc = 0.0
        best_config  = {}
        for group_name, gdata in groups.items():
            for strat in ["soft_voting", "weighted_voting", "stacking"]:
                if strat in gdata:
                    a = gdata[strat].get("auc", 0)
                    if a > best_ens_auc:
                        best_ens_auc = a
                        best_config = {
                            "strategy": strat,
                            "group": group_name,
                            "models": gdata.get("models", []),
                        }

        summary[dataset] = {
            "best_individual_auc": round(best_ind_auc, 4),
            "best_ensemble_auc":   round(best_ens_auc, 4),
            "improvement":         round(best_ens_auc - best_ind_auc, 4),
            "best_config":         best_config,
        }
    return summary


def build_rq2_summary(all_results):
    wins = {"soft_voting": 0, "weighted_voting": 0, "stacking": 0}
    counts = {"soft_voting": [], "weighted_voting": [], "stacking": []}
    for dataset, groups in all_results.items():
        for group_name, gdata in groups.items():
            aucs = {
                s: gdata[s]["auc"]
                for s in ["soft_voting", "weighted_voting", "stacking"]
                if s in gdata
            }
            if not aucs:
                continue
            for s, a in aucs.items():
                counts[s].append(a)
            best = max(aucs, key=aucs.get)
            wins[best] += 1
    return {
        "wins_per_strategy": wins,
        "avg_auc_per_strategy": {
            s: round(float(np.mean(v)), 4) if v else None
            for s, v in counts.items()
        },
    }


def build_rq3_summary(all_results):
    group_stats = {}
    for dataset, groups in all_results.items():
        for group_name, gdata in groups.items():
            if group_name not in group_stats:
                group_stats[group_name] = {
                    "soft_aucs": [], "weighted_aucs": [], "stacking_aucs": [],
                    "avg_diversity": [],
                }
            gs = group_stats[group_name]
            for s, key in [("soft_voting","soft_aucs"),
                            ("weighted_voting","weighted_aucs"),
                            ("stacking","stacking_aucs")]:
                if s in gdata:
                    gs[key].append(gdata[s]["auc"])
            # Mean pairwise diversity (lower correlation = more diverse)
            div = gdata.get("diversity", {})
            vals = [
                div[m1][m2]
                for m1 in div for m2 in div[m1] if m1 != m2
            ]
            if vals:
                gs["avg_diversity"].append(float(np.mean(vals)))

    return {
        g: {
            "avg_soft_auc":      round(float(np.mean(v["soft_aucs"])), 4)      if v["soft_aucs"]      else None,
            "avg_weighted_auc":  round(float(np.mean(v["weighted_aucs"])), 4)  if v["weighted_aucs"]  else None,
            "avg_stacking_auc":  round(float(np.mean(v["stacking_aucs"])), 4)  if v["stacking_aucs"]  else None,
            "avg_diversity_corr":round(float(np.mean(v["avg_diversity"])), 4)  if v["avg_diversity"]  else None,
        }
        for g, v in group_stats.items()
    }



# MAIN


def main():
    args = docopt(__doc__)

    datasets    = args["--datasets"].split(",")
    n_runs      = int(args["--n_runs"])
    strategy    = args["--strategy"].lower()
    preds_dir   = args["--preds_dir"]
    results_dir = args["--results_dir"]

    strategies = {"all", strategy}   # always include the chosen strategy

    all_results = {}   # {dataset: {group_name: group_result}}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset : {dataset}")
        print(f"{'='*60}")

        # Discover which models actually have saved predictions
        ds_preds_path = os.path.join(preds_dir, dataset)
        if not os.path.isdir(ds_preds_path):
            print(f"  [skip] no saved_preds directory for {dataset}")
            continue

        available_models = [
            m for m in ALL_MODELS
            if os.path.exists(os.path.join(
                ds_preds_path, f"{m}_run0_test_preds.npy"
            ))
        ]
        print(f"  Models with predictions: {available_models}")

        if len(available_models) < 2:
            print(f"  [skip] fewer than 2 models available")
            continue

        groups, individual_aucs = build_groups(results_dir, dataset, available_models)

        all_results[dataset] = {}

        for group_name, group_models in groups.items():
            result = run_group(
                group_name, group_models, dataset,
                preds_dir, results_dir, n_runs,
                individual_aucs, strategies,
            )
            if result is not None:
                all_results[dataset][group_name] = result

    if not any(all_results.values()):
        print("\nNo valid ensemble results were generated. Existing summaries were not updated.")
        print("Regenerate prediction files with n_runs.py so every model has predictions, labels, and IDs.")
        return

    # ── Summary files─
    print(f"\n{'='*60}")
    print("  Generating summary files …")
    print(f"{'='*60}")

    summary_dir = os.path.join(results_dir, "summary")

    # Full raw dump
    save_json(all_results, summary_dir, "all_ensemble_results.json")

    # RQ1 – ensemble vs individual
    rq1 = build_rq1_summary(all_results, results_dir)
    save_json(rq1, summary_dir, "RQ1_ensemble_vs_individual.json")
    print("\n  RQ1 – Ensemble vs Individual AUC:")
    for ds, v in rq1.items():
        sign = "+" if v["improvement"] >= 0 else ""
        print(f"    {ds}: best_ind={v['best_individual_auc']:.4f}  "
              f"best_ens={v['best_ensemble_auc']:.4f}  "
              f"Δ={sign}{v['improvement']:.4f}  "
              f"({v['best_config'].get('strategy','?')} / {v['best_config'].get('group','?')})")

    # RQ2 – strategy comparison
    rq2 = build_rq2_summary(all_results)
    save_json(rq2, summary_dir, "RQ2_strategy_comparison.json")
    print("\n  RQ2 – Strategy comparison:")
    print(f"    Wins  : {rq2['wins_per_strategy']}")
    print(f"    Avg AUC: {rq2['avg_auc_per_strategy']}")

    # RQ3 – group diversity analysis
    rq3 = build_rq3_summary(all_results)
    save_json(rq3, summary_dir, "RQ3_diversity_analysis.json")
    print("\n  RQ3 – Group analysis:")
    for g, v in rq3.items():
        print(f"    {g}: soft={v['avg_soft_auc']}  "
              f"weighted={v['avg_weighted_auc']}  "
              f"stacking={v['avg_stacking_auc']}  "
              f"diversity_corr={v['avg_diversity_corr']}")

    print(f"\nAll results saved under  '{results_dir}/'")
    print("DONE")


if __name__ == "__main__":
    main()
