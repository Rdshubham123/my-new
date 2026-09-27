"""CLI for the final competition pipeline (entity_matching/final.py).

    python run_final.py --team MYTEAM                 # auto-detects /kaggle/input data
    python run_final.py --data-dir dataset --out-dir out --gpu off
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entity_matching.final import FinalConfig, run_final  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=None)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--train-s1-sample", type=int, default=100_000)
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
ap.add_argument("--team", default="team")
ap.add_argument("--no-xgb", action="store_true")
ap.add_argument("--no-cat", action="store_true")
ap.add_argument("--n-jobs", type=int, default=None)
a = ap.parse_args()
run_final(FinalConfig(data_dir=a.data_dir, out_dir=a.out_dir, train_s1_sample=a.train_s1_sample,
                      folds=a.folds, gpu=a.gpu, team_name=a.team, use_xgboost=not a.no_xgb,
                      use_catboost=not a.no_cat, n_jobs=a.n_jobs))
