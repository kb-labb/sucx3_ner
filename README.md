# ner_kram

## ToDo

### Questions

- Which HPO method do we trust/like?
  - Why? Why not the others?
- Which parameters do we vary?
  - Based on what?
- Do we see patterns in good/bad hyperparameter combinations?
- How do we continue when we do not do finetuning but pretraining?

### NER HPO Experiments

- HPO with AHSHA / BOHB / PBT
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

#### Results

##### Development

| Tag Family | Trained on        | HPO Alg | cased | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | new-lower-cased-both |
| ---------- | ----------------- | ------- | ----- | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Original   | uncased           | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Original   | uncased-cased-mix | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | cased             | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | uncased           | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |

##### Test

| Tag Family | Trained on        | HPO Alg | cased | uncased | uncased-cased-mix | uncased-cased-both | ne-lower | ne-lower-cased-mix | new-lower-cased-both |
| ---------- | ----------------- | ------- | ----- | ------- | ----------------- | ------------------ | -------- | ------------------ | -------------------- |
| Original   | cased             | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Original   | uncased           | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Original   | uncased-cased-mix | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | cased             | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | uncased           | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | uncased-cased-mix | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |
| Simple     | uncased-cased-mix | BOHB    | -     | -       | -                 | -                  | -        | -                  | -                    |

#### Successful Hyperparameters

| Tag Family | Trained on        | HPO Alg | learning rate | weight decay | warmup ratio | attention dropout | hidden dropout | batch size | ??? |
| ---------- | ----------------- | ------- | ------------- | ------------ | ------------ | ----------------- | -------------- | ---------- | --- |
| Original   | cased             | BOHB    | -             | -            | -            | -                 | -              | -          | -   |
| Original   | uncased           | BOHB    | -             | -            | -            | -                 | -              | -          | -   |
| Original   | uncased-cased-mix | BOHB    | -             | -            | -            | -                 | -              | -          | -   |
| Simple     | cased             | BOHB    | -             | -            | -            | -                 | -              | -          | -   |
| Simple     | uncased           | BOHB    | -             | -            | -            | -                 | -              | -          | -   |
| Simple     | uncased-cased-mix | BOHB    | -             | -            | -            | -                 | -              | -          | -   |

### Potentially Interesting Papers

- [Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better](https://arxiv.org/abs/2106.08962)
- [An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models](https://arxiv.org/abs/2106.09204)
- [Hyper-Parameter Optimization: A Review of Algorithms and Applications](https://arxiv.org/abs/2003.05689)
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html)
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
