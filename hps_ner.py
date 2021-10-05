import torch
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB, ASHAScheduler
from ray.tune import CLIReporter
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import BestRun


def hyperparameter_tune(trainer: Trainer, training_args: TrainingArguments, data_args) -> BestRun:

    # Dynamically set eval_steps based on data and batch size for reasonably fine-grained HPS
    train_set_size = len(trainer.train_dataset)
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

    assert data_args.tune_alg in ['BOHB', 'PBT']

    tune_config = {
        # "weight_decay": tune.uniform(0.0, 0.3),
        # "learning_rate": tune.uniform(1e-5, 1e-3),
        "weight_decay": tune.uniform(0.0, 0.1),
        # "learning_rate": tune.uniform(5e-6, 1e-4),
        "learning_rate": tune.uniform(5e-6, 5e-5),
        # "max_steps": -1,  # Use 1 for smoke test, else -1.
    }
    if data_args.tune_alg == 'BOHB':
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            # number of training_iterations (evaluations) to run for each trial, * 2 to allow for grace period
            max_t=max_training_iterations * 2,
            reduction_factor=3,
            stop_last_trials=True,
        )

        search = TuneBOHB(
            # space=config_space,  # If you want to set the space manually
            # max_concurrent=4
        )
    else:  # data_args.tune_alg == 'PBT'
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            # metric="eval_f1",
            # mode="max",
            perturbation_interval=1,  # perturb at every evaluation
            # perturbation_interval=eval_steps,
            hyperparam_mutations={
                # "weight_decay": tune.uniform(0.0, 0.3),
                # "learning_rate": tune.uniform(1e-5, 1e-3),
                "weight_decay": tune_config["weight_decay"],
                "learning_rate": tune_config["learning_rate"]
            },
            # synch=torch.cuda.device_count() < 2  # Use non-recommended synchronized implementation if not multi-gpu
        )

        search = None

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
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
        checkpoint_score_attr="training_iteration",
        stop=None,
        progress_reporter=reporter,
        # local_dir="/home/joey/extra_space/ray_results/",
        # local_dir="/home/joey/code/kb/ner_kram/ray_results/",
        local_dir=data_args.tune_local_dir,
        name=data_args.tune,
        log_to_file=True,
        fail_fast=True
    )

    trainer.args.eval_steps = org_eval_steps
    trainer.args.save_steps = org_save_steps

    return best_run
