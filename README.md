# Multilayer Perceptron

Mandatory part of the 42 project: classify Wisconsin Breast Cancer samples (`B` / `M`) using a neural network implemented from scratch. NumPy handles linear algebra, and Matplotlib plots learning curves. No pretrained ML models or automatic differentiation are used.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

The source dataset is `data.csv`. Each row contains an ID, a diagnosis (`B` or `M`), and 30 numeric features. CSV files may have no header or a header such as `id,diagnosis,...`. An empty trailing column is also accepted. Missing values, unknown labels, and nonnumeric features produce an error. The ID is not used as a feature.

```bash
python split.py data.csv --ratio 0.8 --seed 42
python train.py --train data/train.csv --valid data/validation.csv
python predict.py --model models/model.npz --data data/validation.csv
```

Training opens the learning curves when it finishes. For a server or a run without graphical windows:

```bash
python train.py --train data/train.csv --valid data/validation.csv --no-show
```

Configure the architecture and training parameters:

```bash
python train.py --train data/train.csv --valid data/validation.csv \
  --layers 24 24 24 --epochs 150 --batch-size 16 --learning-rate 0.01 --seed 42
```

`--layers` specifies hidden layers only. The default architecture is `30 -> 24 -> 24 -> 2`. All commands support `--help`. Command-line options let you change paths for the model, plots, predictions, and dataset partitions.

## Pipeline

1. `split.py` shuffles samples within each class and creates training and validation partitions while approximately preserving class proportions. The seed makes the split reproducible. Each sample belongs to exactly one partition.
2. `train.py` computes each feature's mean and standard deviation using training data only. Standardization uses `x_scaled = (x - mean) / std`. Constant features use a divisor of 1. Validation data uses the same fitted parameters.
3. The network learns through mini-batch SGD. Validation data is used only for evaluation, never for gradient computation. Overlapping IDs between training and validation data are rejected.
4. Each epoch reports loss and accuracy on both full partitions after the weight updates.
5. The model and two learning curves are saved: `plots/loss.png` and `plots/accuracy.png`.
6. `predict.py` loads the model and scaling parameters, then writes IDs, actual and predicted classes, and both class probabilities to `data/predictions.csv`. It reports binary cross-entropy and accuracy. This command expects a labeled CSV in the same format because the assignment requires BCE evaluation.

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

Bonus features are not implemented yet. Adam, early stopping, persistent metric history, and additional metrics can be added after verifying the mandatory part.

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
            -> model.step (one batch)
                 -> gradients
                      -> forward
                           -> linear -> relu (hidden layers)
                           -> linear -> softmax (output layer)
                      -> backward
                           -> softmax_cross_entropy_gradient
                           -> weight and bias gradients for each layer
                           -> relu_derivative when propagating to the preceding layer
                 -> update_parameters
       -> model.metrics (training and validation)
            -> forward
            -> cross_entropy_from_logits
            -> accuracy_score
       -> print_epoch_metrics
  -> model.save
  -> plot_curves
```

`backward()` uses activations and probabilities from the corresponding `forward()` call. It computes gradients; `update_parameters()` applies the parameter updates.

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
  -> binary_cross_entropy and accuracy_score
```
