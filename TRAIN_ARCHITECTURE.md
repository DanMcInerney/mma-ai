# MMA AI Training Architecture

**Complete technical documentation of the model training pipeline using AutoGluon.**

---

## Table of Contents

1. [Overview](#overview)
2. [Training Pipeline Flow](#training-pipeline-flow)
3. [Core Components](#core-components)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [AutoGluon Configuration](#autogluon-configuration)
8. [Feature Engineering](#feature-engineering)
9. [Hyperparameters & Tuning](#hyperparameters--tuning)

---

## Overview

### Architecture Summary

The training pipeline uses **AutoGluon** for automated machine learning, training an ensemble of models to predict UFC fight outcomes. The system supports two prediction tasks:

1. **Win Model** - Predicts P(Fighter 1 wins)
2. **Decision Model** - Predicts P(Fight goes to decision)

### Technology Stack

- **ML Framework:** AutoGluon TabularPredictor
- **Data Processing:** pandas, numpy
- **Normalization:** scikit-learn RobustScaler / StandardScaler
- **Metrics:** log_loss (primary), accuracy, precision, recall, F1, Brier score
- **Model Types:** Gradient Boosting (GBM, XGBoost, CatBoost), Neural Networks (TabPFN, TabM, TabICL), MITRA

---

## Training Pipeline Flow

```
┌──────────────────┐
│  Configuration   │ TrainingConfig dataclass
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Data Preparation │ DataPreparation class
├──────────────────┤
│ 1. Load CSV      │ training_data.csv or training_data_dec.csv
│ 2. Filter fights │ start_date, num_fights, split decisions
│ 3. Select feats  │ String-based filtering or explicit list
│ 4. Split data    │ train 85% / test 15%
│ 5. Normalize     │ RobustScaler (default) or Z-score
│ 6. Add weights   │ Recency-based sample weights (optional)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ AutoGluon Setup  │ TabularPredictor initialization
├──────────────────┤
│ - label='y_true' │
│ - eval_metric=   │ 'log_loss'
│   'log_loss'     │
│ - problem_type=  │ 'binary'
│   'binary'       │
│ - sample_weight  │ Optional recency weights
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Training  │ AutoGluon.fit()
├──────────────────┤
│ Preset: extreme  │ → 7 model types, stacking, ensembling
│   OR             │
│ Preset: best     │ → Hyperparameter tuning, GPU support
│                  │
│ Time limit       │ Default: 3000s (50 minutes)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Evaluation     │ Train & Test performance
├──────────────────┤
│ - Accuracy       │
│ - Log Loss       │
│ - Calibration    │ Reliability diagrams
│ - Feat Import.   │ Feature importance (optional)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Save Artifacts   │
├──────────────────┤
│ - model/         │ AutoGluon model directory
│ - evals.txt      │ Performance metrics
│ - feats.txt      │ Feature list
│ - scaler.pkl     │ Normalization scaler
│ - test_preds.csv │ Test set predictions
│ - calib_plot.png │ Calibration curve
│ - model_stats.txt│ Comprehensive stats
└──────────────────┘
```

---

## Core Components

### 1. TrainingConfig ([train.py:23-75](libs/modeling/train.py#L23-L75))

**Purpose:** Configuration dataclass for training parameters

**Key Parameters:**

```python
@dataclass
class TrainingConfig:
    # Model Configuration
    model_type: str              # 'win' or 'decision'
    preset: str                  # 'extreme' or 'best'
    time_limit: int = 3000       # Training time limit (seconds)
    
    # Data Split
    train_size: float = 0.85     # 85% training
    test_size: float = 0.15      # 15% test
    
    # Feature Selection
    features: Optional[List[str]] = None  # Explicit list (overrides filters)
    included_strings: Optional[List[str]] = None  # Must contain these
    excluded_strings: Optional[List[str]] = None  # Must NOT contain these
    required_strings: Optional[List[str]] = None  # Overrides exclusions
    
    # Data Filtering
    start_date: str = '2014-01-01'  # Filter fights before this date
    num_fights: int = 2             # Min fights per fighter
    include_split_dec: bool = False # Include split decisions
    
    # Preprocessing
    normalize: str = 'robust'       # 'robust', 'zscore', or 'none'
    
    # Recency Weighting
    use_recency_weights: bool = False
    decay_rate: float = 0.125       # Exponential decay rate
    
    # Model Types
    included_model_types: Optional[List[str]] = None
    # ['TABPFNV2', 'TABM', 'TABICL', 'GBM', 'XGB', 'CAT', 'MITRA']
    
    # Feature Analysis
    calculate_importance: bool = False
```

**Validation:**
- Checks `model_type` in `['win', 'decision']`
- Checks `preset` in `['extreme', 'best']`
- Validates `train_size + test_size = 1.0`

### 2. FeatureSelector ([train.py:77-144](libs/modeling/train.py#L77-L144))

**Purpose:** String-based feature filtering

**Methods:**

#### `select_features()`
Applies multi-stage filtering:

```python
# Stage 1: included_strings (whitelist)
if included_strings:
    features = [f for f in features if any(s in f for s in included_strings)]

# Stage 2: excluded_strings (blacklist)
if excluded_strings:
    features = [f for f in features if not any(s in f for s in excluded_strings)]

# Stage 3: required_strings (force-include, overrides excluded)
if required_strings:
    features.extend(required_strings)
```

**Example:**
```python
available = ['age_diff', 'reach_diff', 'sig_str_land_dec_avg_diff', 'sig_str_land_total_diff']

selected = FeatureSelector.select_features(
    available,
    included_strings=['dec_avg', 'age', 'reach'],
    excluded_strings=['total'],
    required_strings=['sig_str_land_total_diff']  # Force include despite 'total' exclusion
)

# Result: ['age_diff', 'reach_diff', 'sig_str_land_dec_avg_diff', 'sig_str_land_total_diff']
```

#### `get_features_from_dataframe()`
Extracts feature columns by excluding metadata:

```python
metadata = {
    'fight_id', 'event_id', 'fighter1_id', 'fighter2_id',
    'fighter1_name', 'fighter2_name', 'event_date', 'method',
    'y_true', 'sample_weight', 'groups', 'fighter_dob'
}
features = [col for col in df.columns if col not in metadata]
```

### 3. ModelTrainer ([train.py:147-518](libs/modeling/train.py#L147-L518))

**Purpose:** Orchestrates full training pipeline

**Workflow:**

```python
class ModelTrainer:
    def train(self):
        # 1. Create model directory
        self.model_dir = self._create_model_directory()  # AutogluonModels/ag-<timestamp>-<type>-<preset>
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # 3. Create predictor
        predictor = self._create_predictor()
        
        # 4. Train model
        predictor = self._train_model(predictor, X_train, y_train, X_test, y_test)
        
        # 5. Evaluate
        results = self._evaluate_model(predictor, X_train, y_train, X_test, y_test)
        
        # 6. Save artifacts
        self._save_results(results, X_test, y_test)
        
        return predictor
```

---

## Data Preparation

### DataPreparation Class ([data_preparation.py](libs/modeling/data_preparation.py))

**Purpose:** Handles all preprocessing before training

### Pipeline Steps

#### 1. **Load Data**

```python
df = pd.read_csv(data_path)  # training_data.csv or training_data_dec.csv
```

**Data Format:**
```
fight_id | fighter1_id | fighter2_id | event_date |
age_diff | reach_diff | sig_str_land_dec_avg_diff | ... |
y_true | result
```

#### 2. **Filter Fights**

```python
# Remove invalid fights
df = df[df['result'].isin(['win', 'loss'])]  # No draws, no contests

# Date filtering
df = df[df['event_date'] >= start_date]  # e.g., >= 2014-01-01

# Fighter experience filtering
fighter_fight_counts = df.groupby(['fighter1_id']).size()
valid_fighters = fighter_fight_counts[fighter_fight_counts >= num_fights].index
df = df[df['fighter1_id'].isin(valid_fighters)]

# Split decision filtering (decision model only)
if not include_split_dec:
    df = df[~df['method'].str.contains('Split Decision', na=False)]
```

#### 3. **Feature Selection**

```python
# Option A: Explicit feature list
if config.features:
    X = df[config.features]

# Option B: String-based filtering
else:
    features = FeatureSelector.select_features(
        available=df.columns,
        included_strings=config.included_strings,
        excluded_strings=config.excluded_strings,
        required_strings=config.required_strings
    )
    X = df[features]

y = df['y_true']  # Target variable
```

#### 4. **Train/Test Split**

```python
# Simple chronological split (no shuffle)
split_idx = int(len(df) * train_size)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]
```

**Why chronological?**
- Prevents data leakage (no future knowledge in training)
- Realistic evaluation (test on future fights)
- Preserves temporal dependencies

#### 5. **Normalization**

**RobustScaler (Default):**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

**Formula:**
```
X_scaled = (X - median) / IQR
```
Where IQR = Interquartile Range (75th percentile - 25th percentile)

**Why RobustScaler?**
- Robust to outliers (uses median/IQR vs mean/std)
- MMA stats have heavy tails (rare KOs, long fights)

**Alternative: Z-Score Normalization**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = (X - mean) / std
```

#### 6. **Recency Weighting (Optional)**

**Purpose:** Give more importance to recent fights

**Formula:**
```python
# Calculate fight age in years
time_since_fight = (latest_date - event_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)

# Exponential decay
weight = np.exp(-decay_rate * time_since_fight)

# Add as sample_weight column
X_train['sample_weight'] = weight
```

**Default `decay_rate = 0.125`:**
- 1 year old fight → weight = 0.88
- 2 years old → weight = 0.78
- 5 years old → weight = 0.54

**Usage in AutoGluon:**
```python
predictor = TabularPredictor(
    sample_weight='sample_weight',
    weight_evaluation=False  # Don't weight evaluation metrics
)
```

---

## Model Training

### AutoGluon TabularPredictor

**Initialization:**
```python
predictor = TabularPredictor(
    label='y_true',                    # Target column
    eval_metric='log_loss',            # Optimization metric
    problem_type='binary',             # Binary classification
    path=model_dir,                    # Save directory
    verbosity=2,                       # Logging level
    sample_weight='sample_weight',     # Recency weights (if enabled)
    weight_evaluation=False            # Don't weight test metrics
)
```

### Training Presets

#### **Preset: 'extreme'**

**Purpose:** Maximize accuracy with heavy computation

**Configuration:**
```python
predictor.fit(
    train_data=train_data,              # Combined X_train + y_train
    presets='extreme',                  # Best possible quality
    time_limit=3000,                    # 50 minutes
    ag_args_fit={'shuffle': False},     # Preserve chronological order
    included_model_types=[              # Model types to train
        'TABPFNV2',  # TabPFN V2 (transformer-based)
        'TABM',      # TabM (attention-based)
        'TABICL',    # TabICL (in-context learning)
        'GBM',       # LightGBM
        'XGB',       # XGBoost
        'CAT',       # CatBoost
        'MITRA'      # MITRA (multi-resolution transformer)
    ]
)
```

**What AutoGluon Does:**
1. Trains 7 base model types
2. Creates stacked ensembles (meta-learners)
3. Weighted ensemble of best models
4. Multi-layer stacking (L1, L2, L3)
5. Hyperparameter search per model type

**Typical Ensemble Structure:**
```
WeightedEnsemble_L3  (final prediction)
  ├─ WeightedEnsemble_L2
  │   ├─ TABPFNV2
  │   ├─ TABM  
  │   ├─ XGBoost
  │   └─ CatBoost
  ├─ GBM
  └─ MITRA
```

#### **Preset: 'best'**

**Purpose:** Balance quality and speed with hyperparameter tuning

**Configuration:**
```python
predictor.fit(
    train_data=train_data,
    tuning_data=test_data,              # Use test set for validation during tuning
    presets='best',                     # Quality-focused with tuning
    excluded_model_types=['KNN'],       # Skip K-Nearest Neighbors
    time_limit=3000,
    use_bag_holdout=True,               # Bag models with holdout validation
    ag_args_fit={
        'stopping_metric': 'log_loss',  # Early stopping criterion
        'num_gpus': 1,                  # GPU acceleration
        'shuffle': False                # Preserve order
    },
    ag_args_ensemble={
        'use_orig_features': False,     # Don't include raw features in ensemble
        'max_base_models': 15,          # Max models in ensemble
        'max_base_models_per_type': 3,  # Max per model family
        'fold_fitting_strategy': 'sequential_local'
    }
)
```

**Differences from 'extreme':**
- Includes hyperparameter tuning
- Uses test set as tuning_data (validation)
- GPU support enabled
- Ensemble size limits for efficiency

---

## Model Evaluation

### Metrics Calculated

```python
def _evaluate_model(predictor, X_train, y_train, X_test, y_test):
    train_scores = predictor.evaluate(train_data)
    test_scores = predictor.evaluate(test_data)
    
    return {
        'accuracy': Proportion of correct predictions
        'log_loss': -log(P(correct class))  # Primary metric
        'precision': TP / (TP + FP)
        'recall': TP / (TP + FN)
        'f1': Harmonic mean of precision/recall
        'brier_score': Mean squared error of probabilities
    }
```

### Log Loss (Primary Metric)

**Formula:**
```python
log_loss = -1/N * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
```

**Why log loss?**
- Penalizes confident wrong predictions heavily
- Calibration-aware (rewards probability accuracy)
- Differentiable (smooth optimization)

**Interpretation:**
- **0.6** - Good UFC prediction performance
- **0.55** - Excellent
- **< 0.5** - Exceptional

### Ensemble Composition

**Printed during evaluation:**
```
Ensemble Composition:
  XGBoost_BAG_L1/S1F1: 0.421
  CatBoost_BAG_L1/S1F1: 0.309
  TABPFNV2_BAG_L1/S1F1: 0.187
  LightGBM_BAG_L1/S1F1: 0.083
```

**Interpretation:**
- Weighted average of base models
- XGBoost carries most weight (42.1%)
- Ensemble diversifies risk

### Feature Importance (Optional)

**Calculation:**
```python
importance = predictor.feature_importance(train_data)
```

**Methods Used:**
- **Permutation importance** - How much does performance drop when feature is shuffled?
- **SHAP values** - Shapley additive explanations
- **Tree-based importance** - Split counts & gains (for GBM/XGB/CAT)

**Example Output:**
```
Top 10 Most Important Features:
  1. age_diff: 0.0234
  2. reach_diff: 0.0198
  3. sig_str_land_dec_avg_diff: 0.0176
  4. td_land_dec_adjperf_diff: 0.0143
  5. ufcage_diff: 0.0121
  ...
```

### Calibration Curve

**Purpose:** Assess probability calibration

**Plot:**
```python
ModelUtils.plot_calibration_curve(n_bins=10)
```

**What it shows:**
- X-axis: Predicted probability bins (0-10%, 10-20%, ..., 90-100%)
- Y-axis: Actual win rate in that bin
- Diagonal line = Perfect calibration

**Ideal:** Points lie on diagonal (predicted 60% → actual 60% win rate)

**Common issues:**
- **Above diagonal:** Model underconfident
- **Below diagonal:** Model overconfident

---

## AutoGluon Configuration

### Model Types Explained

#### **TABPFNV2** (Transformer-based Prior-data Fitted Network V2)
- **Architecture:** Transformer with in-context learning
- **Strength:** Few-shot learning, strong on tabular data
- **Speed:** Fast inference
- **GPU:** Recommended

#### **TABM** (Attention-based deep learning)
- **Architecture:** Multi-head attention layers
- **Strength:** Complex feature interactions
- **Speed:** Moderate
- **GPU:** Recommended

#### **TABICL** (In-Context Learning for Tabular Data)
- **Architecture:** Prompt-based learning
- **Strength:** Adapts to data distribution
- **Speed:** Fast
- **GPU:** Recommended

#### **GBM** (LightGBM)
- **Architecture:** Gradient boosted trees
- **Strength:** Fast, memory-efficient, handles missing values
- **Speed:** Very fast
- **CPU/GPU:** Excellent on CPU

#### **XGB** (XGBoost)
- **Architecture:** Gradient boosted trees
- **Strength:** Regularization, cross-validation, feature importance
- **Speed:** Fast
- **CPU/GPU:** Works well on both

#### **CAT** (CatBoost)
- **Architecture:** Gradient boosted trees
- **Strength:** Categorical features, ordered boosting
- **Speed:** Fast
- **CPU/GPU:** Optimized for both

#### **MITRA** (Multi-resolution Transformer)
- **Architecture:** Hierarchical transformer
- **Strength:** Multi-scale feature learning
- **Speed:** Moderate
- **GPU:** Recommended

### Hyperparameters

**AutoGluon Auto-Tunes:**
```python
# XGBoost example hyperparameters
{
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}
```

**User-Controlled:**
- `time_limit` - Total training budget
- `num_bag_folds` - Bagging folds (default: 8)
- `num_stack_levels` - Stacking layers (default: auto)
- `hyperparameters` - Override auto-tuning (advanced)

---

## Feature Engineering

### Win Model Features

**Typical Configuration:**
```python
features = vSeven_testing2  # ~150 features

included_strings = [
    'dec_avg',              # Time-decayed averages
    'age', 'reach',         # Static features
    'ufcage',               # UFC experience
    'odds',                 # Betting odds
    'days_since_last_fight',
    'time_sec',             # Fight duration priors
    'weightclass_encoded',
    'sig_str_land_total'    # Cumulative stats
]
```

**Feature Categories:**
1. **Static:** `age_diff`, `reach_diff`, `ufcage_diff`
2. **Time-Decayed:** `sig_str_land_dec_avg_diff`, `td_land_dec_avg_diff`
3. **Adjusted Performance:** `sig_str_land_dec_adjperf_diff`
4. **Odds:** `f1_ip_closing_odds`, `f2_ip_closing_odds`
5. **Totals:** `sig_str_land_total_diff`

### Decision Model Features

**Typical Configuration:**
```python
features = DECISION_TEST_FEATS4  # ~100 features

included_strings = [
    'time_sec',      # Fight duration predictors
    'decision',      # Historical decision rate
    'sub', 'ko',     # Finish types (inverse relationship)
    'kd',            # Knockdown rate
    'win',           # Win rate
    'strikes_att',   # Volume
    'distance_att',  # Range fighting
    'td', 'ctrl',    # Grappling
    'weightclass_encoded'
]

include_split_dec = True  # Include split decisions (more data)
```

**Key Insight:** Decision model uses different features than win model:
- More emphasis on **fight pace** (strikes_att, distance_att)
- Less emphasis on **opponent-adjusted stats**

---

## Hyperparameters & Tuning

### Recommended Settings

#### **Development/Testing**
```python
config = TrainingConfig(
    preset='extreme',
    time_limit=1500,  # 25 minutes
    included_model_types=['GBM', 'XGB', 'CAT'],  # Fast models only
    calculate_importance=True
)
```

#### **Production - Win Model**
```python
config = TrainingConfig(
    model_type='win',
    preset='extreme',
    time_limit=3000,  # 50 minutes
    included_model_types=['TABPFNV2', 'TABM', 'TABICL', 'GBM', 'XGB', 'CAT', 'MITRA'],
    use_recency_weights=True,
    decay_rate=0.125,
    normalize='robust'
)
```

#### **Production - Decision Model**
```python
config = TrainingConfig(
    model_type='decision',
    preset='extreme',
    time_limit=3000,
    include_split_dec=True,  # More training data
    use_recency_weights=True,
    normalize='robust'
)
```

### Tuning Tips

1. **Time Limit:**
   - `< 1000s` - Quick iteration, basic ensemble
   - `1000-2000s` - Good quality, multi-layer stacking
   - `> 3000s` - Production quality, full search

2. **Recency Weights:**
   - **Enable** if using data from 2014+
   - **Disable** if fighter evolution isn't expected
   - `decay_rate=0.125` works well (half-life ~5.5 years)

3. **Normalization:**
   - **RobustScaler** - Default, handles outliers
   - **StandardScaler** - If data is normally distributed
   - **None** - Only for tree-based models (they're scale-invariant)

4. **Feature Selection:**
   - Start with explicit `features` list (faster iteration)
   - Use string filters for experimentation
   - Monitor feature importance to prune redundant features

5. **Model Types:**
   - **GPU available:** Include TABPFNV2, TABM, TABICL, MITRA
   - **CPU only:** Focus on GBM, XGB, CAT
   - **Fast iteration:** `['GBM', 'XGB']`

---

## Saved Artifacts

### Model Directory Structure

```
AutogluonModels/ag-20251122_184500-win-extreme/
├── models/                      # AutoGluon model files
│   ├── WeightedEnsemble_L3/
│   ├── XGBoost_BAG_L1/
│   ├── CatBoost_BAG_L1/
│   └── ...
├── evals.txt                    # Performance metrics
├── feats.txt                    # Feature list
├── test_predictions.csv         # Test set predictions
├── training_data.csv            # Processed training data
├── scaler.pkl                   # Normalization scaler
├── model_stats.txt              # Comprehensive statistics
├── calibration_curve.png        # Calibration plot
└── utils_ensembles.pkl          # AutoGluon metadata
```

### evals.txt

```
Model Performance:
Training accuracy: 0.7234
Training log loss: 0.5512
Test accuracy: 0.6823
Test log loss: 0.6145

Best Model: WeightedEnsemble_L3

Top Most Important Features:
1. age_diff: 0.0234
2. reach_diff: 0.0198
...

Configuration:
Model Type: win
Preset: extreme
Time Limit: 3000
...
```

### test_predictions.csv

```
fighter1_name,fighter2_name,y_pred_proba,y_pred,y_true,event_date
Jon Jones,Stipe Miocic,0.7234,1,1,2024-11-16
...
```

---

## Usage Examples

### Basic Win Model Training

```python
from libs.modeling.train import TrainingConfig, ModelTrainer

config = TrainingConfig(
    model_type='win',
    preset='extreme',
    time_limit=3000
)

trainer = ModelTrainer(config)
predictor = trainer.train()
```

### Custom Feature Selection

```python
config = TrainingConfig(
    model_type='win',
    preset='extreme',
    included_strings=['dec_avg', 'age', 'reach', 'odds'],
    excluded_strings=['total', 'mad'],
    required_strings=['sig_str_land_total_diff']  # Force include despite 'total' exclusion
)

trainer = ModelTrainer(config)
predictor = trainer.train()
```

### Decision Model with Recency Weights

```python
config = TrainingConfig(
    model_type='decision',
    preset='extreme',
    include_split_dec=True,
    use_recency_weights=True,
    decay_rate=0.125
)

trainer = ModelTrainer(config)
predictor = trainer.train()
```

### Fast Development Training

```python
config = TrainingConfig(
    model_type='win',
    preset='extreme',
    time_limit=1500,
    included_model_types=['GBM', 'XGB', 'CAT'],  # Fast models only
    calculate_importance=True
)

trainer = ModelTrainer(config)
predictor = trainer.train()
```

---

## Troubleshooting

**Q: Training is too slow**  
A: Reduce `time_limit`, use `included_model_types=['GBM', 'XGB']`, or use `preset='best'` instead of `'extreme'`

**Q: Test accuracy << training accuracy (overfitting)**  
A: Enable `use_recency_weights=True`, increase `num_fights` filter, or reduce feature count

**Q: Log loss is high (> 0.7)**  
A: Check feature quality, ensure proper normalization, verify no data leakage, increase `time_limit`

**Q: GPU not being used**  
A: Ensure `included_model_types` includes GPU models (TABPFNV2, TABM, etc.) and `preset='best'` with `num_gpus=1`

**Q: Feature importance calculation fails**  
A: Some model types don't support permutation importance. Try `included_model_types=['GBM', 'XGB', 'CAT']`

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-22  
**AutoGluon Version:** 1.x  
**Training Script:** [libs/modeling/train.py](libs/modeling/train.py)
