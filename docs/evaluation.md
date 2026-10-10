# Evaluation evidence and reproducibility

## Reported result

The team's final presentation (slide 25) reports 95.9% classification accuracy. The following confusion matrix is transcribed from that slide; rows are true labels and columns are predicted labels.

| True / predicted | Neutral | Left | Right | Long blink | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 61 | 0 | 0 | 0 | 61 |
| Left | 0 | 39 | 0 | 1 | 40 |
| Right | 0 | 1 | 23 | 1 | 25 |
| Long blink | 0 | 5 | 0 | 66 | 71 |

The matrix has 197 examples and 189 correct predictions: `189 / 197 = 95.939%`, rounding to **95.9%**. This checks the arithmetic of the supplied evidence; it is not a new experiment. See [the source slide](reported-results.png) and [final presentation](final-presentation.pdf).

## What was verified for this release

The supplied model loads with scikit-learn 1.8.0. It is a `StandardScaler -> MLPClassifier` pipeline with 209 input features, hidden layers `(96, 48)`, ReLU activation, Adam solver, and `max_iter=1800`. The scaler records 647 fitted samples. This value must not be interpreted as the full dataset size, test size, or participant count.

Model structure, finite probability outputs, shared preprocessing, malformed-input rejection, and training/inference compatibility are covered by hardware-free tests. See `validation.json` for the local test outcome.

## Limits of reproduction

The ZIP does not include the complete segmented training/test dataset, sample-level predictions, or original split manifest. The original training script uses a stratified 80/20 sample split, but its architecture `(64, 32)` and `max_iter=800` differ from the supplied model. Consequently, neither that script nor the archived four continuous streams establish the exact procedure used to obtain the presentation's result.

The released trainer matches the saved model's architecture and applies scaling **after splitting**, fitted only on training data. It uses a stratified 80/20 sample split with seed 42. Re-training on another dataset does not reproduce 95.9% by definition. Samples from the same participant may appear in both partitions; this procedure is not a subject-independent evaluation.

The presentation estimates 0.6-1.0 seconds for eye-to-command latency. This has not been independently measured in this release. The demo documents prototype behavior; it is not clinical validation or an unseen-user benchmark.
