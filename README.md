# ner_kram

## ToDo

- add comparative baselines: `stanza`, `spacy`

### Questions

- Which HPO method do we trust/like?
  - Why? Why not the others?
- Which parameters do we vary?
  - Based on what?
- Do we see patterns in good/bad hyperparameter combinations?
- How do we continue when we do not do finetuning but pretraining?

### NER HPO Experiments

- HPO with ASHA / BOHB / PBT
  - original tags
    - cased
    - uncased
    - uncased-cased-mix
  - simple tags
    - cased
    - uncased
    - uncased-cased-mix
  - evaluate each model on for their respective tag family
    - cased
    - uncased
    - uncased-cased-mix
    - uncased-cased-both
    - ne-lower
    - ne-lower-cased-mix
    - ne-lower-cased-both
  - look at what hyperparameters and combination lead to good results

#### HPO Method

**Motivation for ASHA**: Authors demonstrate slight advantage over BOHB, even commonly on par with PBT. Huggingface provides an example for HPO with ray tune and BO + ASHA (https://huggingface.co/blog/ray-tune). Furthermore, ASHA is used in [https://arxiv.org/pdf/2106.09204.pdf] and seems to be a practical solution due to the native support of a grace period parameter. Potentially, PBT would provide a better performing model at the cost of having to rely on hyperparameter schedules and depressingly long HPO runs with several TB of disk usage. Perhaps we could overcome some of these issues by tinkering with the PBT parameters, but for now it does not seem to be worth the pain.

BO (TPE) + ASHA, 50 trials with a grace period of 1 epoch with the follow hyperparameter space, inspired by [https://arxiv.org/pdf/2106.09204.pdf]:

```
learning_rate: ~U(6e-6, 5e-6)
weight_decay: ~U(0.0, 0.2)
warmup_ratio: ~U(0.0, 0.12)
attention_probs_dropout_prob: ~U(0, 0.2)
hidden_dropout_prob: ~U(0, 0.2)
per_device_train_batch_size: ~Choice([16, 32, 64])
```

**Motivation for RS instead of ASHA**<br>
Early stopping terminates a lot of trials with promising converged performance but allows for exploring a larger number of trials. Random Search with no BO results in a straight-forward analysis of the hyperparameter space. A large hyperparameter space with many hyperparameters requires more time to find good local optimas and is prone to overfitting on the dev-set, and also entails more complex results and prohibits an intuitive understanding of the hyperparameters and their relationships. Therefore, we reduced the search space to:

```
learning_rate: ~Choice([2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5])
weight_decay: ~Choice([0.0, 0.05, 0.10, 0.15])
warmup_ratio: ~Choice([0.0, 0.04, 0.08, 0.12])
```
The learning rate space is more fine-grained due to its importance.

**Why 50 trials?**<br>
A run with 50 trials took ~6h with 4 GPUs on Vega, longer than this might be infeasible. A longer HPO run could be performed for the final model to get the most out of the fine-tuning. To get a sense of the impact of the number of trials, two HPO runs with BO + ASHA is done with 27 trials and 50 trials:

Doubling the time for the HPO run resulted in an f1-dev and f1-test increase of ~0.005 for uncased and ~0.02 for cased. The hypothesis is that this performance increase will not continue linearly with a larger number of trials, and too extensive HPO is prone to overfitting the dev set and thus not increase the test set performance. Therefore, due to the diminising returns, settling for 50 trials seems reasonable. Below are results for 27 trials vs 50 trials:

Development:
| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | uncased           | ASHA-27 | 0.77931| 0.87012 | 0.82533           | 0.82621            | 0.87044  | 0.82607            | 0.82640              |
| Original   | uncased           | ASHA-50 | 0.80012| 0.87619 | 0.83333           | 0.83911            | 0.87444  | 0.83302            | 0.83825              |

Test:
| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | uncased           | ASHA-27 | 0.78353| 0.86388 | 0.82469           | 0.82493            | 0.86206  | 0.82550            | 0.82406              |
| Original   | uncased           | ASHA-50 | 0.80281| 0.86625 | 0.83296           | 0.83526            | 0.86453  | 0.83294            | 0.83444              |


**When switching to pure RS, we lowered the number of trials to 30.**

#### Results

##### Temporary Results For Reference

Trained & evaluated on uncased with default hyperparameters:

|         | BS=16  | BS=32  | BS=64      | BS=128 | BS=256 |
|---------|--------|--------|------------|--------|--------|
| F1-Dev  | 0.8651 | 0.8676 | **0.8683** | 0.8635 | 0.8572 |
| F1-Test | 0.8668 | 0.8663 | **0.867**  | 0.8624 | 0.8556 |

PBT, 20 trials, perturbation interval 1, uncased training & evaluation:<br>
F1-Dev=0.8684 (does not look very promising with the current settings as it took ~13h with 4 gpus)


###### Baselines
Each column illustrates one setting of tag & case type with batch size 64 and original hyperparameters.

|         | org/cased | org/uncased | org/mixed | simple/cased | simple/uncased | simple/mixed |
|---------|-------------|-----------|-----------|----------------|--------------|--------------|
| F1-Dev  | 0.8901      | 0.8683    | 0.866     | 0.9359         | 0.912        | 0.9111       |
| F1-Test | 0.8901      | 0.867     | 0.8687    | 0.9346         | 0.9017       | 0.9118       |

##### Development

| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | RS      | 0.8951 | 0.4067  | 0.7054            | 0.6987             | 0.4110   | 0.7045             | 0.6985               |
| Original   | uncased           | RS      | 0.7847 | 0.8713  | 0.8278            | 0.8293             | 0.8695   | 0.8263             | 0.8285               |
| Original   | uncased-cased-mix | RS      | 0.8821 | 0.8573  | 0.8702            | 0.8698             | 0.8504   | 0.8671             | 0.866                |
| Simple     | cased             | RS      | 0.9345 | 0.3037  | 0.7035            | 0.6974             | 0.3038   | 0.6995             | 0.6941               |
| Simple     | uncased           | RS      | 0.8361 | 0.9157  | 0.8753            | 0.8774             | 0.9154   | 0.8754             | 0.8773               |
| Simple     | uncased-cased-mix | RS      | 0.9275 | 0.9078  | 0.9185            | 0.9177             | 0.9029   | 0.9155             | 0.9153               |

##### Test

| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | RS      | 0.8978 | 0.4053  | 0.6940            | 0.7000             | 0.4103   | 0.6924             | 0.6998               |
| Original   | uncased           | RS      | 0.7811 | 0.8656  | 0.8248            | 0.8245             | 0.8649   | 0.8245             | 0.8242               |
| Original   | uncased-cased-mix | RS      | 0.8833 | 0.8523  | 0.8661            | 0.8680             | 0.8489   | 0.8650             | 0.8663               |
| Simple     | cased             | RS      | 0.9304 | 0.2963  | 0.6940            | 0.6929             | 0.2902   | 0.6879             | 0.6861               |
| Simple     | uncased           | RS      | 0.8299 | 0.9075  | 0.8687            | 0.8702             | 0.9074   | 0.8685             | 0.8702               |
| Simple     | uncased-cased-mix | RS      | 0.9219 | 0.8988  | 0.9083            | 0.9104             | 0.8950   | 0.9064             | 0.9085               |

#### Successful Hyperparameters

| Tag Family | Trained on        | HPO Alg | learning rate | weight decay | warmup ratio |
| ---------- | ----------------- | ------- | ------------- | ------------ | ------------ |
| Original   | cased             | RS      | 7e-05         | 0.15         | 0.04         |
| Original   | uncased           | RS      | 5e-05         | 0.10         | 0.08         |
| Original   | uncased-cased-mix | RS      | 8e-05         | 0.15         | 0.12         |
| Simple     | cased             | RS      | 5e-05         | 0.05         | 0.04         |
| Simple     | uncased           | RS      | 8e-05         | 0.05         | 0.04         |
| Simple     | uncased-cased-mix | RS      | 6e-05         | 0.05         | 0.12         |


###### Baselines Comparison
Each column illustrates one setting of tag & case type with the performance difference after HPO: `F1(HPO) - F1(baseline)`

|         | org/cased | org/uncased | org/mixed | simple/cased | simple/uncased | simple/mixed |
|---------|-------------|-----------|-----------|----------------|--------------|--------------|
| F1-Dev  | +0.005     | +0.003    | +0.0042    | -0.0014        | +0.0037      | +0.0074      |
| F1-Test | +0.0077    | -0.0014   | -0.0026    | -0.0042        | +0.0058      | -0.0035      |

#### Old ASHA Results

##### Development (Old)

| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | ASHA    | 0.89669| 0.46699 | 0.72543           | 0.71970            | 0.46980  | 0.72487            | 0.71877              |
| Original   | uncased           | ASHA    | 0.77931| 0.87012 | 0.82533           | 0.82621            | 0.87044  | 0.82607            | 0.82640              |
| Original   | uncased-cased-mix | ASHA    | 0.88407| 0.85457 | 0.86897           | 0.86949            | 0.84911  | 0.86641            | 0.86681              |

##### Test (Old)

| Tag Family | Trained on        | HPO Alg | cased  | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | ne-lower-cased-both  |
| ---------- | ----------------- | ------- | -----  | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | ASHA    | 0.90315| 0.47993 | 0.72141           | 0.72828            | 0.48020  | 0.71946            | 0.72648              |
| Original   | uncased           | ASHA    | 0.78353| 0.86388 | 0.82469           | 0.82493            | 0.86206  | 0.82550            | 0.82406              |
| Original   | uncased-cased-mix | ASHA    | 0.88791| 0.85132 | 0.86593           | 0.86980            | 0.84549  | 0.86198            | 0.86694              |

#### Successful Hyperparameters (Old)

| Tag Family | Trained on        | HPO Alg | learning rate | weight decay | warmup ratio | attention dropout | hidden dropout | batch size |
| ---------- | ----------------- | ------- | ------------- | ------------ | ------------ | ----------------- | -------------- | ---------- |
| Original   | cased             | ASHA    | 4.462279e-05  | 0.049931167  | 0.049692212  | 0.061813731       | 0.131402719    | 32         |
| Original   | uncased           | ASHA    | 1.994873e-05  | 0.040402108  | 0.058761397  | 0.191723976       | 0.089677054    | 16         |
| Original   | uncased-cased-mix | ASHA    | 4.567007e-05  | 0.130067974  | 0.018670453  | 0.018893785       | 0.182467284    | 64         |


### Potentially Interesting Papers

- [Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better](https://arxiv.org/abs/2106.08962)
- [An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models](https://arxiv.org/abs/2106.09204)
- [Hyper-Parameter Optimization: A Review of Algorithms and Applications](https://arxiv.org/abs/2003.05689)
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html)
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
- [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934)
- [Deepspeed: Curriculum Learning](https://www.deepspeed.ai/tutorials/curriculum-learning/)
