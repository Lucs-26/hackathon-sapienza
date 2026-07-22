# Dream TIM — Machine Unlearning

PyTorch MLP for multi-label classification with **machine unlearning**: given a pre-trained model and a `forget_data.csv` subset, remove that subset's influence while preserving accuracy.

## Structure

```
main.py                  # end-to-end pipeline
utils/
  model.py                # DynamicMLP
  functions.py            # load_pickle, prepare_data
  unlearning.py           # ssd, fisher_forgetting, gradient_ascent, finetune_retain
  eval.py                  # precision@k, MIA AUC, final_score, save_submission
data/                     # partitioned dataset + pre-trained model_artifact
```

## Pipeline

1. Split data into retain / forget / validation (validation carved only from retain).
2. Reload the pre-trained `DynamicMLP` from `model_artifact`.
3. Unlearn the forget set with **SSD** (`unlearning.ssd`): dampens weights whose Fisher information is disproportionately tied to the forget set vs. the retain set.
4. Write the submission (`model_artifact`, `execution_time.txt`, `validation_ids.csv`) via `save_submission`.

## Score (`eval.py`)

- 45% Precision@10 (accuracy)
- 45% MIA resistance — `1 - 2·|AUC - 0.5|`, ideal attacker AUC = 0.5
- 10% Execution time (penalty above 300s)

## Run

```bash
unzip data.zip
pip install -r requirements.txt   # or: conda env create -f environment.yml
python main.py
```
