import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cast.config import CASTConfig
from cast.train import CASTTrainer
from cast.data.dataset import load_dataset, prepare_splits, preprocess_splits
from cast.utils.seed import set_seed
from cast.utils.metrics import print_metrics


ALL_EXPERIMENTS = [
    {"dataset": "retail", "task": None},
    {"dataset": "retail", "task": "churn"},
    {"dataset": "cdnow", "task": None},
    {"dataset": "cdnow", "task": "churn"},
    {"dataset": "instacart", "task": None},
    {"dataset": "instacart", "task": "churn"},
    {"dataset": "sales_weekly", "task": None},
    {"dataset": "sales_weekly", "task": "seasonality"},
    {"dataset": "tafeng", "task": None},
    {"dataset": "tafeng", "task": "repurchase"},
]

BENCHMARK_SEEDS = [42, 123, 456]

ABLATION_VARIANTS = [
    {"id": "Full", "name": "Full CAST",          "flags": {}},
    {"id": "A1",   "name": "w/o Causal Disc.",   "flags": {"ablate_no_causal": True}},
    {"id": "A2",   "name": "w/o Hyperbolic",     "flags": {"ablate_no_hyperbolic": True}},
    {"id": "A3",   "name": "w/o Sparsity Gate",  "flags": {"ablate_no_sparsity_gate": True}},
    {"id": "A4",   "name": "w/o Causal Attn",    "flags": {"ablate_no_causal_attn": True}},
    {"id": "A5",   "name": "w/o Focal Loss",     "flags": {"ablate_no_focal": True}},
]

def _baseline_worker(args_tuple):
    dataset, task, seed, model_names, data_root = args_tuple
    import time as _time
    from cast.utils.seed import set_seed as _set_seed
    from cast.data.dataset import load_dataset as _load, prepare_splits as _split
    from cast.models.baselines import (
        ALL_BASELINES as _ALL, _NNBaseline)

    _set_seed(seed)
    label = f"{dataset}/{task}" if task else dataset

    X, y = _load(dataset, data_root=data_root, task=task)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, seed=seed)

    if model_names:
        name_set = set(model_names)
        targets = [cls for cls in _ALL if cls.name in name_set]
    else:
        targets = list(_ALL)

    results = []
    for cls in targets:
        try:
            model = cls()
            t0 = _time.time()
            if isinstance(model, _NNBaseline):
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
            elapsed = _time.time() - t0

            train_m = model.evaluate(X_train, y_train)
            val_m = model.evaluate(X_val, y_val)
            test_m = model.evaluate(X_test, y_test)

            r = {
                "model": model.name, "label": label, "seed": seed,
                "elapsed": round(elapsed, 2),
                "train": train_m, "val": val_m, "test": test_m,
            }
            results.append(r)
            print(f"  [done] {model.name} | {label} | seed={seed} | "
                  f"F1={test_m['f1']:.4f} | {elapsed:.1f}s")
        except Exception as _e:
            print(f"  [FAIL] {cls.name} | {label} | seed={seed}: {_e}")

    return results


def load_dataset_config(dataset, task, config_dir="configs"):
    yaml_path = CASTConfig.resolve_yaml_path(dataset, task, config_dir)
    if os.path.exists(yaml_path):
        import yaml
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        return overrides
    return {}


def run_single(dataset, task, args, config_overrides=None):
    set_seed(args.seed)

    label = f"{dataset}/{task}" if task else dataset
    print("\n" + "=" * 60)
    print(f"CAST | {label} | KAN={args.kan_type}")
    print("=" * 60)

    X, y = load_dataset(dataset, data_root=args.data_root, task=task)
    print(f"Data: {X.shape}, pos_ratio={y.mean():.4f}, "
          f"sparsity={(X == 0).mean() if X.ndim == 2 else (X.reshape(X.shape[0], -1) == 0).mean():.4f}")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
        X, y, seed=args.seed
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    X_train_pp, X_val_pp, X_test_pp, m_train, m_val, m_test = preprocess_splits(
        X_train, X_val, X_test
    )

    raw_ch = X.shape[-1] if X.ndim == 3 else 1
    config_dir = getattr(args, 'config_dir', 'configs')

    if hasattr(args, 'config') and args.config is not None:
        yaml_path = args.config
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, "r") as f:
                yaml_overrides = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file {yaml_path} not found, using defaults")
            yaml_overrides = {}
    else:
        yaml_overrides = load_dataset_config(dataset, task, config_dir)

    config = CASTConfig(
        seed=args.seed,
        seq_len=X.shape[1],
        input_channels=X_train_pp.shape[-1],
        raw_channels=raw_ch,
        kan_type=args.kan_type,
    )
    if args.device is not None:
        config.device = args.device

    for k, v in yaml_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    cli_map = {
        "kan_order": args.kan_order, "kan_hidden_dim": args.kan_hidden_dim,
        "kan_num_layers": args.kan_num_layers, "embedding_dim": args.embedding_dim,
        "pretrain_epochs": args.pretrain_epochs, "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "dag_threshold": args.dag_threshold,
    }
    for k, v in cli_map.items():
        if v is not None:
            setattr(config, k, v)

    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    config.__post_init__()

    print(f"Config: T={config.seq_len}, C={config.input_channels}, "
          f"KAN={config.kan_type}(order={config.kan_order}), "
          f"layers={config.kan_num_layers}, d={config.embedding_dim}, "
          f"device={config.device}")

    t0 = time.time()
    trainer = CASTTrainer(config)
    trainer.train(X_train, y_train, X_val, y_val,
                  X_train_pp=X_train_pp, X_val_pp=X_val_pp,
                  masks_train=m_train, masks_val=m_val,
                  skip_phase1=getattr(args, 'skip_phase1', False))
    elapsed = time.time() - t0

    print(f"\n--- Results for {label} (elapsed {elapsed:.1f}s) ---")
    train_metrics = trainer.evaluate(X_train_pp, y_train, masks=m_train, apply_transform=False)
    print_metrics(train_metrics, prefix="[Train] ")
    val_metrics = trainer.evaluate(X_val_pp, y_val, masks=m_val, apply_transform=False)
    print_metrics(val_metrics, prefix="[Val] ")
    test_metrics = trainer.evaluate(X_test_pp, y_test, masks=m_test, apply_transform=False)
    print_metrics(test_metrics, prefix="[Test] ")

    if args.save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kan_type = config.kan_type
        seed = config.seed
        save_file = f"{args.save_path}/{dataset}_{task or 'default'}_{kan_type}_{timestamp}_seed{seed}.pt"
        os.makedirs(args.save_path, exist_ok=True)
        trainer.save(save_file)
        print(f"Model saved to {save_file}")

    return {
        "dataset": dataset,
        "task": task,
        "label": label,
        "n_samples": len(X),
        "seq_len": X.shape[1],
        "elapsed": round(elapsed, 1),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="CAST: Causal-Aware Sparse Temporal Enhancement")
    parser.add_argument("--dataset", type=str, default="merchant",
                        choices=["merchant", "retail", "cdnow", "instacart", "sales_weekly", "tafeng"])
    parser.add_argument("--industry", type=str, default=None,
                        choices=[None, "Industry-0", "Industry-1", "Industry-2", "Industry-3"])
    parser.add_argument("--task", type=str, default=None,
                        choices=[None, "churn", "seasonality", "repurchase"])
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 14 experiments (4 merchant industries + 5 datasets x 2 tasks)")
    parser.add_argument("--kan_type", type=str, default="taylor",
                        choices=["taylor", "bspline", "fourier",
                                 "chebykan", "jacobikan", "rbfkan",
                                 "hermitekan", "waveletkan", "legendrekan"])
    parser.add_argument("--kan_order", type=int, default=None, help="Override YAML")
    parser.add_argument("--kan_hidden_dim", type=int, default=None, help="Override YAML")
    parser.add_argument("--kan_num_layers", type=int, default=None, help="Override YAML")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Override YAML")
    parser.add_argument("--pretrain_epochs", type=int, default=None, help="Override YAML")
    parser.add_argument("--finetune_epochs", type=int, default=None, help="Override YAML")
    parser.add_argument("--batch_size", type=int, default=None, help="Override YAML")
    parser.add_argument("--lr", type=float, default=None, help="Override YAML")
    parser.add_argument("--dag_threshold", type=float, default=None, help="Override YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Ablation: skip Phase 1 causal discovery")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study: Full + 5 variants, multi-seed")
    parser.add_argument("--ablation_variants", nargs="*", default=None,
                        help="Subset of variant IDs, e.g. Full A1 A2")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run all baselines (statistical + neural)")
    parser.add_argument("--baseline_model", nargs="*", default=None,
                        help="Run only specified baseline(s), e.g. --baseline_model UniTS Mamba")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Parallel workers for --run_baselines (1=sequential)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark: all models x all datasets x 3 seeds, report mean±std")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory containing per-dataset YAML configs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config YAML file (overrides auto-detection)")
    args = parser.parse_args()

    if args.ablation:
        from collections import defaultdict

        save_dir = args.save_path or "results/ablation"
        os.makedirs(save_dir, exist_ok=True)

        experiments = [e for e in ALL_EXPERIMENTS if e["dataset"] != "merchant"]
        if args.dataset != "merchant":
            experiments = [e for e in experiments if e["dataset"] == args.dataset]
            if args.task is not None:
                experiments = [e for e in experiments if e["task"] == args.task]

        variants = ABLATION_VARIANTS
        if args.ablation_variants:
            v_set = set(args.ablation_variants)
            variants = [v for v in variants if v["id"] in v_set]

        print(f"Ablation: {len(experiments)} datasets x {len(variants)} variants "
              f"x {len(BENCHMARK_SEEDS)} seeds = "
              f"{len(experiments) * len(variants) * len(BENCHMARK_SEEDS)} runs")

        all_results = []

        for exp in experiments:
            ds, tsk = exp["dataset"], exp["task"]
            label = f"{ds}/{tsk}" if tsk else ds
            config_dir = getattr(args, 'config_dir', 'configs')

            samples_info = ""
            try:
                X_tmp, _ = load_dataset(ds, data_root=args.data_root, task=tsk)
                samples_info = f"  (samples={len(X_tmp)}, T={X_tmp.shape[1]})"
            except Exception:
                pass

            print(f"\n{'#'*70}")
            print(f"# ABLATION: {label}{samples_info}")
            print(f"{'#'*70}")

            for variant in variants:
                vid, vname, vflags = variant["id"], variant["name"], variant["flags"]
                seed_f1s = []

                for seed in BENCHMARK_SEEDS:
                    overrides = {**vflags}

                    args.seed = seed
                    set_seed(seed)

                    print(f"\n--- [{vid}] {vname} | {label} | seed={seed} ---")
                    cast_result = run_single(ds, tsk, args, config_overrides=overrides)
                    cast_result["model"] = vname
                    cast_result["variant_id"] = vid
                    cast_result["seed"] = seed
                    all_results.append(cast_result)
                    seed_f1s.append(cast_result["test"]["f1"])

                mean_f1 = np.mean(seed_f1s)
                std_f1 = np.std(seed_f1s)
                print(f"  >> {vname} | {label}: F1 = {mean_f1:.4f} \u00b1 {std_f1:.4f}")

        groups = defaultdict(list)
        for r in all_results:
            groups[(r["variant_id"], r["label"])].append(r["test"])

        ds_labels = [f"{e['dataset']}/{e['task']}" if e["task"] else e["dataset"]
                     for e in experiments]

        print(f"\n{'='*100}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'='*100}")
        pm = "\u00b1"
        hdr = f"{'Variant':<24} {'Dataset':<22} {'F1 (mean'+pm+'std)':>16} {'AUC':>16} {'AUPRC':>16}"
        print(hdr)
        print("-" * 100)
        for label in ds_labels:
            for v in variants:
                key = (v["id"], label)
                if key not in groups:
                    continue
                mlist = groups[key]
                f1s = np.array([m["f1"] for m in mlist])
                aucs = np.array([m.get("auc", 0) for m in mlist])
                auprcs = np.array([m.get("auprc", 0) for m in mlist])
                print(f"{v['name']:<24} {label:<22} "
                      f"{f1s.mean():.4f}\u00b1{f1s.std():.4f}  "
                      f"{aucs.mean():.4f}\u00b1{aucs.std():.4f}  "
                      f"{auprcs.mean():.4f}\u00b1{auprcs.std():.4f}")
            print()

        print("F1 DELTA vs FULL (negative = component removal hurts)")
        print("-" * 100)
        for label in ds_labels:
            full_key = ("Full", label)
            if full_key not in groups:
                continue
            full_f1 = np.mean([m["f1"] for m in groups[full_key]])
            for v in variants:
                if v["id"] == "Full":
                    continue
                key = (v["id"], label)
                if key not in groups:
                    continue
                abl_f1 = np.mean([m["f1"] for m in groups[key]])
                delta = abl_f1 - full_f1
                marker = "\u2193" if delta < -0.001 else ("\u2191" if delta > 0.001 else "\u2248")
                print(f"  {v['name']:<24} {label:<22} {delta:>+8.4f}  {marker}")
            print()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_dir, f"ablation_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {json_path}")
        return all_results

    if args.benchmark:
        from cast.models.baselines import run_all_baselines
        from collections import defaultdict

        save_dir = args.save_path or "results"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(save_dir, f"benchmark_{timestamp}.jsonl")

        all_results = []

        experiments = ALL_EXPERIMENTS
        if args.dataset != "merchant":
            experiments = [e for e in experiments if e["dataset"] == args.dataset]
            if args.task is not None:
                experiments = [e for e in experiments if e["task"] == args.task]

        for exp in experiments:
            ds, tsk = exp["dataset"], exp["task"]
            label = f"{ds}/{tsk}" if tsk else ds
            for seed in BENCHMARK_SEEDS:
                print(f"\n{'#'*70}")
                print(f"# Benchmark: {label} | seed={seed}")
                print(f"{'#'*70}")

                args.seed = seed
                set_seed(seed)

                X, y = load_dataset(ds, data_root=args.data_root, task=tsk)
                X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
                    X, y, seed=seed)

                print(f"Data: {X.shape}, pos={y.mean():.4f}, "
                      f"split: {len(X_train)}/{len(X_val)}/{len(X_test)}")

                bl_results = run_all_baselines(
                    X_train, y_train, X_val, y_val, X_test, y_test, label=label)
                for r in bl_results:
                    r["seed"] = seed
                    all_results.append(r)
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(r, default=str) + "\n")

                set_seed(seed)

                cast_result = run_single(ds, tsk, args)
                cast_result["model"] = "CAST"
                cast_result["seed"] = seed
                all_results.append(cast_result)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(cast_result, default=str) + "\n")

        groups = defaultdict(list)
        for r in all_results:
            groups[(r["model"], r["label"])].append(r["test"])

        models_order = ["Croston", "SBA", "ZIP", "Hurdle",
                        "DSN", "SoftShape", "TimeMIL", "CAST"]
        labels_order = [
            f"{e['dataset']}/{e['task']}" if e["task"] else e["dataset"]
            for e in ALL_EXPERIMENTS]

        print(f"\n{'='*105}")
        print(f"BENCHMARK SUMMARY  (seeds: {BENCHMARK_SEEDS})")
        print(f"{'='*105}")
        hdr = f"{'Model':<12} {'Dataset':<26} {'F1 (mean±std)':>16} {'AUC':>16} {'AUPRC':>16}"
        print(hdr)
        print("-" * 105)
        for label in labels_order:
            for model in models_order:
                key = (model, label)
                if key not in groups:
                    continue
                mlist = groups[key]
                f1s = np.array([m["f1"] for m in mlist])
                aucs = np.array([m.get("auc", 0) for m in mlist])
                auprcs = np.array([m.get("auprc", 0) for m in mlist])
                f1_s = f"{f1s.mean():.4f}±{f1s.std():.4f}"
                auc_s = f"{aucs.mean():.4f}±{aucs.std():.4f}"
                auprc_s = f"{auprcs.mean():.4f}±{auprcs.std():.4f}"
                print(f"{model:<12} {label:<26} {f1_s:>16} {auc_s:>16} {auprc_s:>16}")
            print()

        json_path = os.path.join(save_dir, f"benchmark_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Full results saved to {json_path}")
        print(f"Incremental log: {jsonl_path}")
        return all_results

    if args.run_baselines:
        from collections import defaultdict
        experiments = ALL_EXPERIMENTS if args.run_all else None
        if experiments is None:
            experiments = [{"dataset": args.dataset, "task": args.task}]

        model_filter = args.baseline_model
        max_w = args.max_workers

        save_dir = args.save_path or "results"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_tag = args.dataset if not args.run_all else "all"
        bl_tag = "_".join(model_filter) if model_filter else "all"
        jsonl_path = os.path.join(
            save_dir, f"baselines_{bl_tag}_{ds_tag}_{timestamp}.jsonl")

        work_items = []
        for exp in experiments:
            for seed in BENCHMARK_SEEDS:
                work_items.append((
                    exp["dataset"], exp["task"], seed,
                    model_filter, args.data_root))

        n_jobs = len(work_items)
        print(f"\nBaseline run: {n_jobs} jobs "
              f"({len(experiments)} datasets × {len(BENCHMARK_SEEDS)} seeds), "
              f"models={model_filter or 'ALL'}, max_workers={max_w}")

        all_baseline_results = []

        if max_w > 1:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = _mp.get_context("spawn")
            effective_w = min(max_w, n_jobs)
            print(f"Launching {effective_w} parallel workers (spawn) …")

            with ProcessPoolExecutor(max_workers=effective_w,
                                     mp_context=ctx) as pool:
                futures = {pool.submit(_baseline_worker, item): item
                           for item in work_items}
                for fut in as_completed(futures):
                    try:
                        results = fut.result()
                    except Exception as e:
                        ds, tsk, seed, _, _ = futures[fut]
                        print(f"  [FAIL] {ds}/{tsk} seed={seed}: {e}")
                        continue
                    for r in results:
                        all_baseline_results.append(r)
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(r, default=str) + "\n")
        else:
            for item in work_items:
                try:
                    results = _baseline_worker(item)
                except Exception as e:
                    print(f"  [FAIL] {item[0]}/{item[1]} seed={item[2]}: {e}")
                    continue
                for r in results:
                    all_baseline_results.append(r)
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(r, default=str) + "\n")

        if not all_baseline_results:
            print("No results collected.")
            return []

        groups = defaultdict(list)
        for r in all_baseline_results:
            groups[(r["model"], r["label"])].append(r["test"])

        models_seen = list(dict.fromkeys(
            r["model"] for r in all_baseline_results))
        labels_seen = list(dict.fromkeys(
            r["label"] for r in all_baseline_results))

        print(f"\n{'='*100}")
        print(f"BASELINE SUMMARY  (seeds: {BENCHMARK_SEEDS})")
        print(f"{'='*100}")
        print(f"{'Model':<12} {'Dataset':<26} {'F1 (mean±std)':>18} "
              f"{'AUC':>18} {'AUPRC':>18}")
        print("-" * 100)
        for label in labels_seen:
            for model in models_seen:
                key = (model, label)
                if key not in groups:
                    continue
                mlist = groups[key]
                f1s    = np.array([m["f1"] for m in mlist])
                aucs   = np.array([m.get("auc", 0) for m in mlist])
                auprcs = np.array([m.get("auprc", 0) for m in mlist])
                print(f"{model:<12} {label:<26} "
                      f"{f1s.mean():.4f}±{f1s.std():.4f}  "
                      f"{aucs.mean():.4f}±{aucs.std():.4f}  "
                      f"{auprcs.mean():.4f}±{auprcs.std():.4f}")
            print()

        json_path = os.path.join(
            save_dir, f"baselines_{bl_tag}_{ds_tag}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(all_baseline_results, f, indent=2, default=str)
        print(f"Saved to {json_path}  |  incremental: {jsonl_path}")
        return all_baseline_results

    if args.run_all:
        all_results = []
        for exp in ALL_EXPERIMENTS:
            result = run_single(exp["dataset"], exp["task"], args)
            all_results.append(result)

        print("\n" + "=" * 70)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("=" * 70)
        header = f"{'Dataset':<25} {'Acc':>7} {'F1':>7} {'AUC':>7} {'AUPRC':>7} {'Time':>7}"
        print(header)
        print("-" * 70)
        for r in all_results:
            t = r["test"]
            print(f"{r['label']:<25} {t['accuracy']:>7.4f} {t['f1']:>7.4f} "
                  f"{t.get('auc', 0):>7.4f} {t.get('auprc', 0):>7.4f} {r['elapsed']:>6.1f}s")

        if args.save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(args.save_path, f"summary_{timestamp}.json")
            master_log = os.path.join(args.save_path, "experiment_log.jsonl")

            os.makedirs(args.save_path, exist_ok=True)

            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            experiment_entry = {
                "timestamp": timestamp,
                "kan_type": args.kan_type,
                "seed": args.seed,
                "total_experiments": len(all_results),
                "avg_f1": sum(r["test"]["f1"] for r in all_results) / len(all_results),
                "avg_auc": sum(r["test"].get("auc", 0) for r in all_results) / len(all_results),
                "total_time": sum(r["elapsed"] for r in all_results),
                "summary_file": f"summary_{timestamp}.json"
            }

            with open(master_log, "a") as f:
                f.write(json.dumps(experiment_entry, default=str) + "\n")

            print(f"\nSummary saved to {summary_path}")
            print(f"Experiment logged to {master_log}")

        return all_results

    else:
        task = args.task
        if args.dataset == "merchant":
            if args.industry is None:
                args.industry = "Industry-0"
                print(f"Merchant dataset requires --industry. Defaulting to {args.industry}")
            task = args.industry

        result = run_single(args.dataset, task, args)
        return result


if __name__ == "__main__":
    main()
