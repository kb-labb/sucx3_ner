import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB, ASHAScheduler
from datasets import load_dataset, load_metric
from ray.tune import CLIReporter
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = load_dataset('glue', 'mrpc')
metric = load_metric('glue', 'mrpc')

def encode(examples):
    outputs = tokenizer(
        examples['sentence1'], examples['sentence2'], truncation=True)
    return outputs

encoded_dataset = dataset.map(encode, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    mets = metric.compute(predictions=predictions, references=labels)
    # tune.report(score={'eval_f1': mets['f1']})
    # tune.report(eval_f1=mets['f1'])
    return mets


# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
bs = 16
save_eval_steps = 300
training_args = TrainingArguments(
    "test", evaluation_strategy="steps", eval_steps=save_eval_steps, save_strategy="steps", save_steps=save_eval_steps,
    disable_tqdm=False, num_train_epochs=3, per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2, fp16=True)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

tune_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    "learning_rate": tune.uniform(1e-5, 1e-3),
    # "max_steps": 1000,  # Use 1 for smoke test, else -1.
}

reporter = CLIReporter(
    parameter_columns={
                    "weight_decay": "w_decay",
                    "learning_rate": "lr",
                    # "per_device_train_batch_size": "train_bs/gpu",
                    # "num_train_epochs": "num_epochs"
                },
    metric_columns=[
                    "eval_accuracy", "eval_loss", "eval_precision", "eval_recall", "eval_f1", "epoch", "training_iteration"
                ])


scheduler = HyperBandForBOHB(
                            time_attr="training_iteration",
                            # metric="eval_f1",
                            # mode="max",
                            # max_t=81,
                            max_t=2,
                            reduction_factor=2,
                            stop_last_trials=True,
                            )

bohb_search = TuneBOHB(
    # metric="eval_f1",
    # mode="max"
    # space=config_space,  # If you want to set the space manually
    # max_concurrent=4
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    direction="maximize",
    metric="eval_f1",
    mode="max",
    backend="ray",
    n_trials=4,  # number of trials
    local_dir="/home/joey/code/kb/ner_kram/ray_results/",
    progress_reporter=reporter,
    name="hps_example",
    scheduler=scheduler,
    search_alg=bohb_search,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",
    fail_fast=True
)

print("***** Best Hyperparameters found *****")
for n, v in best_run.hyperparameters.items():
    if n != 'max_steps':  # Use original max_steps
        # setattr(trainer.args, n, v)
        print(f"{n}: {v}")
