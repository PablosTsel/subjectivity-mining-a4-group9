"""Seed check (Block 4). Trains fBERT and HateBERT again with the same settings but
seeds 1 and 2, to see if the in-domain vs cross-domain results stay the same.
The main seed 42 results are not changed. Run from block4_fbert/: python seed_check/run_seeds.py"""
import sys, shutil, importlib
from pathlib import Path

sys.path[:0] = [str(Path(".").resolve()), str(Path("../BLOCK3").resolve())]
SCRATCH_MODELS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seed_check/models")

for module_name in ["fbert_block4", "hatebert_block3"]:
    mod = importlib.import_module(module_name)
    tag = "fbert" if module_name.startswith("fbert") else "hatebert"
    for seed in [1, 2]:
        mod.SEED = seed
        mod.TRAINING["manual_seed"] = seed
        out = Path(f"seed_check/{tag}_seed{seed}")
        for setting in ["in_domain", "cross_domain"]:
            if (out / f"run_{tag}_{setting}.json").exists():
                continue
            models = SCRATCH_MODELS / f"{tag}_seed{seed}"
            mod.run_setting(setting, ".", str(out), str(models))
            shutil.rmtree(models / setting)  # weights not needed, only predictions/metrics
        mod.compare_results(str(out))
