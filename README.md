# Machine Learning from Scratch

An evolving learning repository for implementing foundational machine-learning
algorithms with NumPy before comparing them with established library versions.
The goal is to make the mathematics and optimisation steps visible rather than
to replace production ML frameworks.

## Current scope

The repository currently contains one worked notebook:

- **Linear regression:** forward prediction, mean squared error, analytical
  gradients, gradient descent, early stopping, feature scaling, and evaluation
  on scikit-learn's diabetes dataset.

The implementation is compared with `sklearn.linear_model.LinearRegression`
using the same train/test split. The saved notebook run reports:

| Model | Training MSE | Test MSE |
| --- | ---: | ---: |
| NumPy implementation | 2739.08 | 3098.91 |
| scikit-learn baseline | 2739.06 | 3100.14 |

These figures document one reproducible learning experiment; they are not a
claim of state-of-the-art performance.

## Setup

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab linear-regression.ipynb
```

Run the notebook from top to bottom to regenerate the model parameters, metrics,
and plots.

## What is implemented from scratch

- Linear prediction with a weight vector and bias
- Mean squared error
- Gradients for weights and bias
- Batch gradient descent
- Tolerance-based early stopping

NumPy provides array operations; scikit-learn is used for the dataset, split,
metrics, and comparison baseline rather than for the custom training loop.

## Limitations and roadmap

- This is deliberately a learning project, not a reusable ML library.
- The implementation currently covers only ordinary linear regression.
- The notebook does not yet include automated tests or repeated-seed analysis.
- Future work may extract the model into a tested Python module before adding
  logistic regression, decision trees, and neural-network fundamentals.
