# Multilayer Perceptron

Mandatory part of the 42 project: classify Wisconsin Breast Cancer samples (`B` / `M`) using a neural network implemented from scratch. NumPy handles linear algebra, and Matplotlib plots learning curves. No pretrained ML models or automatic differentiation are used.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the project

Run all commands below from the repository root with the virtual environment activated. In every new terminal:

```bash
source .venv/bin/activate
```

The source dataset is `data.csv`. Each row contains an ID, a diagnosis (`B` or `M`), and 30 numeric features. CSV files may have no header or a header such as `id,diagnosis,...`. An empty trailing column is also accepted. Missing values, unknown labels, and nonnumeric features produce an error. The ID is not used as a feature.

### 1. Split the dataset

```bash
python split.py data.csv --ratio 0.8 --seed 42 \
  --train data/train.csv --valid data/validation.csv
```

This creates 454 training rows and 115 validation rows for the supplied dataset. Split once and reuse the same partitions when comparing models.

### 2. Train the mandatory baseline

```bash
python train.py --train data/train.csv --valid data/validation.csv
```

Defaults: SGD, two hidden layers of 24 neurons each, 100 epochs, batch size 16, learning rate 0.01, seed 42, and no early stopping. This saves `models/model.npz`, `history/run.json`, `plots/loss.png`, and `plots/accuracy.png`, then opens the plots.

To save plots without opening windows:

```bash
python train.py --train data/train.csv --valid data/validation.csv --no-show
```

### 3. Predict with the baseline model

```bash
python predict.py --model models/model.npz \
  --data data/validation.csv --output data/predictions.csv
```

The output CSV contains IDs, actual classes, predicted classes, and both probabilities. The terminal reports BCE, accuracy, precision, recall, F1, and the confusion matrix. Input must be labeled so these metrics can be computed. Prediction does not update the model.

### 4. Train SGD with separate output files

```bash
python train.py --train data/train.csv --valid data/validation.csv \
  --optimizer sgd --layers 24 24 --epochs 100 --batch-size 16 \
  --learning-rate 0.01 --seed 42 \
  --model models/sgd.npz --history history/sgd.json \
  --plots plots/sgd --no-show
```

### 5. Train Adam with early stopping

```bash
python train.py --train data/train.csv --valid data/validation.csv \
  --optimizer adam --layers 24 24 --epochs 300 --batch-size 16 \
  --learning-rate 0.001 --seed 42 \
  --early-stopping --patience 15 --min-delta 0.0001 \
  --model models/adam.npz --history history/adam.json \
  --plots plots/adam --no-show
```

The epoch count is an upper limit. Early stopping monitors validation loss and restores the actual lowest-loss weights before saving. Remove `--early-stopping` to run all requested epochs and save the final weights. The early stopping flag also works with SGD.

### 6. Predict with either saved model

```bash
python predict.py --model models/sgd.npz \
  --data data/validation.csv --output data/sgd_predictions.csv

python predict.py --model models/adam.npz \
  --data data/validation.csv --output data/adam_predictions.csv
```

### 7. Compare learning curves

After creating both histories in steps 4 and 5:

```bash
python compare.py history/sgd.json history/adam.json \
  --labels SGD Adam --output plots/comparison --no-show
```

This saves `loss.png`, `accuracy.png`, `precision.png`, `recall.png`, and `f1.png` under `plots/comparison/`. Remove `--no-show` to open the charts. You can provide more than two histories, with one unique label per file.

### 8. Customize the architecture

`--layers` specifies hidden layers only; input and output sizes remain 30 and 2. For example, train a network with three hidden layers:

```bash
python train.py --train data/train.csv --valid data/validation.csv \
  --layers 32 16 8 --optimizer sgd --epochs 150 --batch-size 8 \
  --learning-rate 0.01 --seed 42 \
  --model models/custom.npz --history history/custom.json \
  --plots plots/custom --no-show
```

Each training command starts a fresh model. Reusing an output path replaces that run's output, so choose distinct model, history, and plot paths to retain multiple runs.

### 9. Run tests

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Run only the core or bonus tests:

```bash
python -m unittest discover -s tests -p 'test_core.py' -v
python -m unittest discover -s tests -p 'test_bonus.py' -v
```

### 10. Show command-line help

```bash
python split.py --help
python train.py --help
python predict.py --help
python compare.py --help
```

### Command options

| Command | Option | Default / meaning |
| --- | --- | --- |
| `split.py` | `dataset` | Required positional input CSV path |
| `split.py` | `--ratio` | `0.8`; fraction assigned to training |
| `split.py` | `--seed` | `42`; reproducible partitioning |
| `split.py` | `--train`, `--valid` | `data/train.csv`, `data/validation.csv`; output paths |
| `train.py` | `--train`, `--valid` | Required input partition paths |
| `train.py` | `--layers` / `--layer` | `24 24`; hidden-layer sizes |
| `train.py` | `--epochs` | `100`; maximum training epochs |
| `train.py` | `--batch-size` / `--batch_size` | `16`; examples per parameter update |
| `train.py` | `--learning-rate` / `--learning_rate` | `0.01`; update step multiplier, also when selecting Adam |
| `train.py` | `--optimizer` | `sgd`; choose `sgd` or `adam` |
| `train.py` | `--seed` | `42`; weight initialization and batch ordering |
| `train.py` | `--early-stopping` | Disabled unless supplied |
| `train.py` | `--patience` | `10`; epochs without sufficient validation improvement |
| `train.py` | `--min-delta` | `0.0`; minimum improvement to reset patience |
| `train.py` | `--model` | `models/model.npz`; saved model path |
| `train.py` | `--history` | `history/run.json`; metric history path |
| `train.py` | `--plots` | `plots`; learning curve directory |
| `train.py`, `compare.py` | `--no-show` | Save charts without opening windows |
| `predict.py` | `--data` | Required labeled input CSV |
| `predict.py` | `--model` | `models/model.npz`; model to load |
| `predict.py` | `--output` | `data/predictions.csv`; prediction CSV path |
| `compare.py` | `histories` | At least two positional JSON history paths |
| `compare.py` | `--labels` | History filename stems; labels must be unique |
| `compare.py` | `--output` | `plots/comparison`; chart directory |


## Pipeline

1. `split.py` shuffles samples within each class and creates training and validation partitions while approximately preserving class proportions. The seed makes the split reproducible. Each sample belongs to exactly one partition.
2. `train.py` computes each feature's mean and standard deviation using training data only. Standardization uses `x_scaled = (x - mean) / std`. Constant features use a divisor of 1. Validation data uses the same fitted parameters.
3. The network learns through mini-batch SGD or optional Adam. Validation data is used only for evaluation, never for gradient computation. Overlapping IDs between training and validation data are rejected.
4. Each epoch reports loss, accuracy, precision, recall, and F1 on both full partitions after the weight updates.
5. The model, JSON metric history, and two learning curves are saved: `plots/loss.png` and `plots/accuracy.png`.
6. `predict.py` loads the model and scaling parameters, then writes IDs, actual and predicted classes, and both class probabilities to `data/predictions.csv`. It reports binary cross-entropy, accuracy, precision, recall, F1, and the confusion matrix. This command expects a labeled CSV in the same format because the assignment requires BCE evaluation.

The model file, `models/model.npz`, contains the format version, architecture, weights, biases, scaling parameters, and class order `[B, M]`. Loading does not use pickle. Ties between class probabilities are resolved as B.

## How the network works

For a batch of N samples, input `A0` has shape `(N, 30)`. Each layer's weight matrix has shape `(number_of_inputs, number_of_outputs)`, and its bias has shape `(1, number_of_outputs)`.

**Forward propagation:** each layer computes `Z = A @ W + b`. Hidden layers apply `ReLU(Z) = max(0, Z)`. The output layer applies softmax: `p_k = exp(z_k - max(z)) / sum(exp(z_j - max(z)))`. Subtracting the maximum prevents exponential overflow. Weights are initialized randomly with scale `sqrt(2 / fan_in)`, and biases start at zero.

**Loss:** training uses categorical cross-entropy, computed stably from logits: `mean(logsumexp(z) - z_correct)`. For two classes, this is equivalent to BCE: `-mean(y*log(P(M)) + (1-y)*log(1-P(M)))`. The prediction program explicitly computes BCE, clipping probabilities away from 0 and 1 to protect the logarithms.

**Backpropagation:** the gradient of loss with respect to output logits is `(P - Y_one_hot) / N`. For each layer, `dW = A_previous.T @ dZ` and `db = sum(dZ, axis=0)`. The gradient for the preceding hidden layer is `(dZ @ W.T) * (A_previous > 0)`, applying the chain rule and the ReLU derivative. At zero, the chosen ReLU derivative is 0.

**Gradient descent:** after computing all gradients for a batch, update `W -= learning_rate*dW`, and update biases similarly. One epoch is one pass through all training samples. Samples are shuffled before each epoch. The final partial batch is also processed.

**Accuracy:** the fraction of samples whose most probable class matches the actual label. Decreasing training loss alongside increasing validation loss can indicate overfitting.

## Project structure

- `src/data.py`: CSV handling, stratified splitting, and standardization.
- `src/network.py`: network, linear transformation, ReLU, softmax, losses, forward/backward passes, SGD, model saving and loading.
- `src/visualization.py`: plotting and saving learning curves.
- `split.py`, `train.py`, `predict.py`: three command-line programs.
- `tests/test_core.py`: mathematical and behavioral checks.
- `tests/test_bonus.py`: optimizer, early stopping, history, metrics, and CLI checks.
- `src/optimizers.py`, `src/early_stopping.py`, `src/history.py`, `src/metrics.py`: bonus components.
- `compare.py`: comparison charts for saved runs.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests check analytical gradients for all parameters of a small network against central finite differences, loss reduction on synthetic data, save/load prediction equivalence, softmax stability, agreement between BCE and training loss, dataset splitting, and CSV/constant-feature handling.

Verified run on the supplied `data.csv`, using seed 42, an 80/20 split, the default architecture, and 100 epochs:

| Partition | Samples | Loss | Accuracy |
| --- | ---: | ---: | ---: |
| Train | 454 | 0.034163 | 99.12% |
| Validation | 115 | 0.058594 | 97.39% |

BCE after reloading the model: `0.058594`. These results use one fixed validation split, rather than an independent test set. The final digits may vary with the numerical environment.

## Bonus features

All five suggested bonus categories are implemented from scratch using NumPy:

- **Adam optimizer** (`src/optimizers.py`): `--optimizer adam` enables exponential moving averages of gradients and squared gradients, bias correction, and adaptive updates. Defaults: beta1=0.9, beta2=0.999, epsilon=1e-8. One optimizer instance retains its state across all batches of a run. The default optimizer remains SGD. Set the learning rate explicitly when selecting Adam; `0.001` is used in the example below.
- **Persistent history** (`src/history.py`): every training run saves JSON with settings, every epoch's train/validation metrics, and a summary of the saved model. Use `--history` to choose a path; the default is `history/run.json`. Use distinct paths for runs you want to compare.
- **Early stopping** (`src/early_stopping.py`): enable `--early-stopping`. Training stops after `--patience` consecutive epochs without an improvement larger than `--min-delta` relative to the last significant improvement. Defaults: patience=10 and min_delta=0.0. The actual lowest-validation-loss parameters are retained independently of this threshold and restored before saving, including when the epoch limit is reached. Without this flag, the final epoch is saved.
- **Additional metrics** (`src/metrics.py`): precision, recall, F1, and confusion counts are recorded for train and validation each epoch. Precision/recall/F1 are printed each epoch. The final validation report and `predict.py` also print the confusion matrix. M is the positive class. A zero denominator yields a score of 0. The matrix uses actual classes as rows and predicted classes as columns in B/M order: `[[TN, FP], [FN, TP]]`.
- **Multiple-model curves** (`compare.py`): overlay train/validation curves from two or more saved histories. The command saves five charts: loss, accuracy, precision, recall, and F1. Each run has one color, with dashed training and solid validation curves. Runs may have different lengths after early stopping. For a meaningful comparison, use the same train/validation split.

### Verified bonus run

The commands in [Running the project](#running-the-project) cover training both optimizers, early stopping, predictions, and model comparison.

On the provided data and existing seed-42 split, the Adam example stopped after 40 epochs and restored epoch 25. Validation accuracy was 97.39%, BCE 0.069433, precision 0.9762, recall 0.9535, and F1 0.9647. Confusion matrix: `[[71, 1], [2, 41]]`. The 100-epoch SGD run retained its previous results: 97.39% accuracy and loss 0.058594. Adam does not guarantee better validation performance than SGD.

### History and saved model semantics

`epochs` contains measurements before any best-weight restoration. `summary.saved_epoch` identifies the weights actually saved, and `summary.train`/`summary.validation` are recomputed after restoration. Confusion counts are included in each epoch's records. Model files retain the original prediction format; Adam's moment buffers are not saved, so the files support inference rather than resuming an optimizer state.

Bonus tests check Adam updates and bias correction, early stopping and independent snapshots, best-weight restoration in the training loop, confusion metrics and zero denominators, JSON round trips, and complete CLI runs through training, prediction, and comparison.

## Reading the execution flow

Start with `main()` in the program you want to study. `create_parser()` defines its arguments, and the remaining calls describe its processing stages.

Splitting (`split.py`):

```text
main -> split_dataset
          -> load_csv -> validate_data_row
          -> split_indices
          -> write_csv (training and validation)
```

Training (`train.py`):

```text
main
  -> validate_training_options
  -> prepare_training_data
       -> load_csv
       -> fit_scaler (training data only)
       -> standardize (training and validation)
  -> Network (initialize random weights and zero biases)
  -> train_model (epoch loop)
       -> train_one_epoch
            -> shuffled_batches
            -> model.train_batch (one batch)
                 -> compute_gradients
                      -> forward
                           -> linear -> relu (hidden layers)
                           -> linear -> softmax (output layer)
                      -> backpropagation
                           -> softmax_cross_entropy_gradient
                           -> weight and bias gradients for each layer
                           -> relu_derivative when propagating to the preceding layer
                 -> sgd_update (SGD) or Adam.update
       -> evaluate_model (training and validation)
            -> forward
            -> cross_entropy_from_logits
            -> classification_metrics
       -> print_epoch_metrics
       -> EarlyStopping.observe (when enabled)
  -> EarlyStopping.restore (when enabled)
  -> evaluate_model (saved weights after optional best-weight restoration)
  -> model.save
  -> save_history
  -> plot_curves
```

`backpropagation()` uses activations and probabilities from the corresponding `forward()` call. It computes gradients; `sgd_update()` or `Adam.update()` applies the parameter updates. `compute_gradients()` combines the forward and backpropagation passes, and `train_batch()` performs one complete batch update. `Network.evaluate()` returns loss and accuracy without changing parameters; `evaluate_model()` also includes the bonus classification metrics.

Prediction (`predict.py`):

```text
main
  -> validate_output_path
  -> load_csv
  -> predict_dataset
       -> Network.load
       -> standardize
       -> predict_probabilities -> forward
  -> save_predictions
  -> binary_cross_entropy and classification_metrics
```
