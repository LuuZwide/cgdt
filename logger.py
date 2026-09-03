"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os
import utils
import wandb
import re


class Logger:
    def __init__(self, variant):

        self.no_wandb = variant["no_wandb"]
        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")


    def log_metrics(self, outputs, iter_num, total_transitions_sampled):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        outputs["evaluation/total_transitions_sampled"] = total_transitions_sampled

        rcsl_tables = {}
        rcsl_metrics = {}
        pattern = re.compile(r"^rcsl_evaluation/")

        # Separate RCSL outputs into tables and metrics
        for k, v in list(outputs.items()):
            if pattern.match(k):
                if isinstance(v, wandb.Table):
                    rcsl_tables[k] = v
                else:
                    rcsl_metrics[k] = v
                del outputs[k] # Remove from original outputs dictionary

        # Update main outputs with RCSL metrics
        outputs.update(rcsl_metrics)
        
        for k, v in outputs.items():
            print(f"{k}: {v}")
        
        for k, v in rcsl_tables.items():
            print(f"{k}: {v}")
            
        for k, v in rcsl_metrics.items():
            print(f"{k}: {v}")

        if self.no_wandb:
            return

        wandb.log(outputs, step=iter_num, commit=False)
        wandb.log(rcsl_tables, step=iter_num, commit=True)
            
    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        seed = variant["seed"]
        return f"{prefix}/{now}-{exp_name}/{seed}"
