# Extended KTBench: Ensemble Learning for Deep Knowledge Tracing

This repository extends the original **KTBench** framework to support **Ensemble Learning** for Deep Knowledge Tracing (DKT). While individual neural architectures often suffer from specific inductive biases, this extended pipeline allows researchers to systematically combine diverse DKT models to achieve more robust, stable, and accurate predictions across different educational contexts.

It supports:
- Training a **single model** or **all models multiple times** (`n_runs`) to extract stable base predictions.
- Aggregating predictions using multiple **Ensemble Strategies** (Voting, Stacking).
- Evaluating the impact of **Architectural Diversity** by grouping distinct families of models.

---

## Key Features

### 1. Ensemble Aggregation Strategies
The framework implements three distinct methods to combine the base models' predictions:
- **Soft Voting:** Unweighted mathematical average of the base probabilities.
- **Weighted Voting:** Average weighted proportionally to each model's standalone AUC performance.
- **Stacking:** A meta-learning approach using a Logistic Regression meta-model trained on a validation set to dynamically correct base model biases.

### 2. Architectural Groups
To evaluate the impact of structural diversity, models can be evaluated across five predefined configurations:
- **RNN Family:** Strictly homogeneous recurrent models (RNN, LSTM, DKT+).
- **Cross-Family:** A trio of diverse baseline paradigms (LSTM, SAKT, DKVMN).
- **Diverse:** Maximum heterogeneity combining recurrent, memory, attention, convolutional (FCN), and latent (KQN) models.
- **Top-3:** Greedily selects the three highest-performing standalone models per dataset.
- **All Models:** Comprehensive aggregation of all available architectures.

---

## Project Structure

- `dataloader.py`: Handles loading and preprocessing of KT datasets (supports one-hot and random projection encodings).
- `evaluation.py`: Contains training, testing, and multi-criteria evaluation metrics (AUC, F1, Accuracy, Time, Size).
- `n_runs.py`: Trains all base models for multiple runs to generate stabilized prediction vectors.
- `ensemble_run.py`: Aggregates the stabilized predictions using the defined strategies and groups.
---

## Usage

### Step 1: Train Base Models (Generate Stabilized Predictions)
Training deep learning models is stochastic. We run each model multiple times (default: 5) to reduce variance and generate stable base estimators.

```bash
python3 n_runs.py --hidden=128 --epochs=10
```
### Step 2: Run Ensemble Evaluation
Once the base models are trained and their predictions are logged, run the ensemble pipeline to evaluate the aggregation strategies across the architectural groups:

```bash
python3 ensemble_run.py
```

## Datasets

After downloading, datasets should be placed in the `dataset/` directory, with one subfolder per dataset (e.g., `dataset/assist2009/`, `dataset/statics/`, etc.). 
Each folder must contain the required CSV files (`builder_train.csv`, `builder_test.csv`) following the specified format.

Once this structure is respected, all scripts will run automatically without additional configuration.

### Dataset Format

Each `.csv` file should follow a three-line structure for each interaction sequence:
```text
<sequence_length>
<question_id_1>,<question_id_2>,...
<correctness_1>,<correctness_2>,...
```

## License

This project is licensed under the [MIT License](Lisence).

## Acknowledgement

This work was supported by the COPCOT project (ANR-22-CE38-0003), funded by the French National Research Agency (ANR).












