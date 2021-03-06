from typing import Optional
import os
import shutil
import json
from datetime import datetime
import wandb

from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB, ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.logger import DEFAULT_LOGGERS
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import BestRun
from pathlib import Path


BEST_TUNED_MODELS_PATH = "./best_tuned_models"


def get_immediate_child_directory_with_sub_name(parent_dir: str, sub_name: str) -> Optional[str]:
    directory_contents = os.listdir(parent_dir)
    for item in directory_contents:
        item_path = parent_dir + "/" + item
        if not os.path.isdir(item_path):
            continue
        if sub_name in item:
            return item

    return None


def save_best_model(best_run: BestRun, tune_name: str, tune_local_dir: str) -> None:
    # Create directories
    Path(BEST_TUNED_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    target_dir = BEST_TUNED_MODELS_PATH + "/" + tune_name
    # Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Save model and hyperparameter configuration
    tune_dir = tune_local_dir + "/" + tune_name

    # Find best run directory
    best_run_dir = get_immediate_child_directory_with_sub_name(tune_dir, best_run.run_id)
    if best_run_dir is None:
        print(f"Could not find best run directory run_id={best_run.run_id}")
        return

    best_run_dir = tune_dir + "/" + best_run_dir
    # Go down two checkpoint levels
    for i in range(2):
        checkpoint_dir = get_immediate_child_directory_with_sub_name(best_run_dir, "checkpoint")
        if best_run_dir is None:
            print(f"Could not find best run checkpoint directory inside: {best_run_dir}")
            return

        best_run_dir += "/" + checkpoint_dir

    target_dir_path = Path(target_dir)
    if target_dir_path.exists() and target_dir_path.is_dir():
        print("Target directory already exists, removing/overriding:", target_dir)
        shutil.rmtree(target_dir_path)

    shutil.copytree(src=best_run_dir, dst=target_dir)
    # Write hyperparameters as json pretty print
    with open(target_dir + "/hyperparameters.json", 'w') as f:
        json.dump(best_run.hyperparameters, f, indent=True)


def hyperparameter_tune(trainer: Trainer, training_args: TrainingArguments, data_args) -> BestRun:
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Dynamically set eval_steps based on data and batch size for reasonably fine-grained HPS
    train_set_size = len(trainer.train_dataset)

    # num_gpus = torch.cuda.device_count()
    # bs = training_args.per_device_train_batch_size * (num_gpus if num_gpus > 0 else 1)
    bs = training_args.per_device_train_batch_size
    opt_steps_per_epoch = train_set_size / bs
    evals_per_epoch = 4
    eval_steps = int(opt_steps_per_epoch / evals_per_epoch)  # evaluate & checkpoint 4 times per epoch

    org_eval_steps = trainer.args.eval_steps
    org_save_steps = trainer.args.save_steps
    trainer.args.eval_steps = eval_steps
    trainer.args.save_steps = eval_steps

    # This corresponds to a full run to num_train_epochs
    max_training_iterations = int(evals_per_epoch * training_args.num_train_epochs)

    # Set this to true, and ray will try to resume execution of a cancelled session in the experiment directory
    resume = False

    assert data_args.tune_alg in ['BOHB', 'PBT', 'ASHA']

    tune_config = {
        # "weight_decay": tune.uniform(0.0, 0.3),
        # "learning_rate": tune.uniform(1e-5, 1e-3),
        "weight_decay": tune.uniform(0.0, 0.2),
        "learning_rate": tune.uniform(5e-6, 5e-5),
        # "learning_rate": tune.uniform(5e-6, 5e-5),
        "warmup_ratio": tune.uniform(0, 0.12),
        "attention_probs_dropout_prob": tune.uniform(0, 0.2),
        "hidden_dropout_prob": tune.uniform(0, 0.2),
        "per_device_train_batch_size": tune.choice([16, 32, 64]),
        # "max_steps": -1,  # Use 1 for smoke test, else -1.
    }

    # time_attr = "training_iteration"
    time_attr = "epoch"

    if data_args.tune_alg == 'BOHB':
        scheduler = HyperBandForBOHB(
            time_attr=time_attr,
            # number of training_iterations (evaluations) to run for each trial, * 2 to allow for grace period
            # max_t=max_training_iterations * 2,
            max_t=int(training_args.num_train_epochs) * 2,
            reduction_factor=3,
            stop_last_trials=True,
        )

        search = TuneBOHB(
            # space=config_space,  # If you want to set the space manually
            max_concurrent=4
        )
    elif data_args.tune_alg == 'ASHA':
        scheduler = ASHAScheduler(
            time_attr=time_attr,
            # number of training_iterations (evaluations) to run for each trial, * 2 to allow for grace period
            # max_t=max_training_iterations,
            max_t=int(training_args.num_train_epochs),
            grace_period=1,  # Give each trial at least an epoch
            # grace_period=int(evals_per_epoch / 2),  # Give each trial at least half an epoch
            reduction_factor=3,
        )
        search = HyperOptSearch()
    else:  # data_args.tune_alg == 'PBT'
        scheduler = PopulationBasedTraining(
            time_attr=time_attr,
            # metric="eval_f1",
            # mode="max",
            # perturbation_interval=1,  # perturb at every evaluation
            perturbation_interval=1 / evals_per_epoch,  # perturb at every evaluation
            # perturbation_interval=eval_steps,
            hyperparam_mutations=tune_config,
            # synch=torch.cuda.device_count() < 2  # Use non-recommended synchronized implementation if not multi-gpu
        )

        search = None

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "warmup_ratio": "wr",
            "attention_probs_dropout_prob": "att_do",
            "hidden_dropout_prob": "hi_do",
            "per_device_train_batch_size": "bs",
            # "per_device_train_batch_size": "train_bs/gpu",
            # "num_train_epochs": "num_epochs"
        },
        metric_columns=[
            "eval_accuracy", "eval_loss", "eval_precision", "eval_recall", "eval_f1",
            "epoch", "training_iteration"
        ])

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        metric="eval_f1",
        mode="max",
        direction="maximize",
        backend="ray",
        n_trials=data_args.tune_trials,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        scheduler=scheduler,
        search_alg=search,
        keep_checkpoints_num=1,
        # checkpoint_score_attr="training_iteration",
        checkpoint_score_attr="epoch",
        stop=None,
        progress_reporter=reporter,
        # local_dir="/home/joey/extra_space/ray_results/",
        # local_dir="/home/joey/code/kb/ner_kram/ray_results/",
        local_dir=data_args.tune_local_dir,
        name=data_args.tune,
        log_to_file=True,
        fail_fast=True,
        resume=resume,
        callbacks=[WandbLoggerCallback(project="ner_kram",
                                       entity="joeyohman",
                                       group=data_args.tune + "_" + dt_string,
                                       # api_key="0fc05c8f0ff7f9219378a081a69de35fc26c1011",
                                       # api_key_file="wandb_key_file.txt",
                                       # log_config=data_args.tune_alg == 'PBT',
                                       # start_method="fork",
                                       settings=wandb.Settings(start_method="fork"),
                                       )],
    )

    trainer.args.eval_steps = org_eval_steps
    trainer.args.save_steps = org_save_steps

    save_best_model(best_run, data_args.tune, data_args.tune_local_dir)

    return best_run
