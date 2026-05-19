# Design Document: Refactored Inference Data Creation System

## Executive Summary

This document outlines the design for refactoring `create_inference_data.py` to support multiple model types (win/loss, decision, KO, SUB) with a highly organized, testable, and extensible architecture following Google engineering best practices.

## Current State Analysis

### Current Architecture
- **Monolithic Class**: Single `CreateInferenceData` class handles all responsibilities
- **Hardcoded Output Format**: Only produces `_diff` columns (win/loss format)
- **Tight Coupling**: Data loading, transformation, and formatting are intertwined
- **Limited Extensibility**: Adding new model types requires modifying core logic

### Current Flow
1. Load static data (age, reach, height, etc.)
2. Update time-dependent stats (age, ufcage, days_since_last_fight)
3. Calculate derived stats (ratios, avgs, dec_avgs)
4. Load dynamic data (fight statistics)
5. Combine static and dynamic
6. Add odds features
7. **Subtract fighter2 from fighter1** (creates only `_diff` columns)
8. Add fight experience
9. Return dictionary of fighter_name -> DataFrame

### Problem Statement
- Decision model requires `fighter1_<stat>`, `fighter2_<stat>`, AND `<stat>_diff` columns
- Current implementation only creates `_diff` columns
- No clear separation of concerns
- Difficult to test individual components
- Adding new model types (KO, SUB) would require significant refactoring

## Proposed Architecture

### Design Principles
1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Generate Everything, Filter Later**: Always generate `fighter1_<stat>`, `fighter2_<stat>`, and `<stat>_diff` columns; filter features in `predict.py` based on model needs
3. **Dependency Injection**: Components receive dependencies rather than creating them
4. **Testability**: Each component can be tested in isolation
5. **Extensibility**: New model types only require feature filtering logic, not new transformers
6. **Immutability**: Data transformations create new objects rather than mutating existing ones

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceDataBuilder                      │
│  (Orchestrator - coordinates the pipeline)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      DataLoader (Abstract Base)        │
        │  - load_fighter_data()                 │
        │  - filter_fighters()                   │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────┐                      ┌──────────────┐
│ StaticLoader │                      │DynamicLoader │
│ - age        │                      │ - sig_str    │
│ - reach      │                      │ - td         │
│ - height     │                      │ - etc.       │
└──────────────┘                      └──────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      StatUpdater (Abstract Base)       │
        │  - update_age()                        │
        │  - update_ufcage()                     │
        │  - update_days_since_last_fight()      │
        │  - update_ratios()                     │
        │  - update_avgs()                       │
        │  - update_dec_avgs()                   │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      FeatureTransformer               │
        │  - transform_features()               │
        │  Always generates:                    │
        │  - fighter1_<stat>                    │
        │  - fighter2_<stat>                    │
        │  - <stat>_diff                        │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      FeatureEnricher (Chain)           │
        │  - add_odds_features()                  │
        │  - add_fight_experience()               │
        │  - add_metadata()                       │
        └───────────────────────────────────────┘
                            │
                            ▼
                    Final DataFrame
```

## Component Design

### 1. InferenceDataBuilder (Orchestrator)

**Responsibility**: Coordinates the entire pipeline

**Interface**:
```python
class InferenceDataBuilder:
    def __init__(
        self,
        csv_path: str,
        fight_list: List[Tuple[str, str, str]],
        bfo_odds: Dict[str, int] = None
    )
    
    def build(self) -> Dict[str, pd.DataFrame]
```

**Responsibilities**:
- Initialize all components
- Execute pipeline in correct order
- Handle errors and validation
- Return final inference data with ALL features (fighter1_*, fighter2_*, *_diff)

**Dependencies**:
- DataLoader
- StatUpdater
- FeatureTransformer (single implementation)
- FeatureEnricher

**Key Design Decision**:
- No `model_type` parameter needed - always generates all feature types
- Feature filtering happens in `predict.py` before model prediction

### 2. DataLoader (Abstract Base Class)

**Responsibility**: Load and filter fighter data from CSV

**Interface**:
```python
class DataLoader(ABC):
    @abstractmethod
    def load_fighter_data(self, fighter_name: str) -> pd.DataFrame
    
    @abstractmethod
    def filter_fighters(self, fight_list: List[Tuple]) -> List[Tuple]
```

**Concrete Implementations**:
- `StaticDataLoader`: Loads static features (age, reach, height, etc.)
- `DynamicDataLoader`: Loads dynamic features (fight statistics)

**Key Methods**:
- `load_fighter_data()`: Load data for a specific fighter
- `filter_fighters()`: Remove fights with insufficient data
- `handle_duplicate_ids()`: Resolve multiple fighter_ids

### 3. StatUpdater

**Responsibility**: Update time-dependent and derived statistics

**Interface**:
```python
class StatUpdater:
    def __init__(self, fighter_dfs: Dict[str, pd.DataFrame])
    
    def update_age(self) -> Dict[str, pd.DataFrame]
    def update_ufcage(self) -> Dict[str, pd.DataFrame]
    def update_days_since_last_fight(self) -> Dict[str, pd.DataFrame]
    def update_ratios(self) -> Dict[str, pd.DataFrame]
    def update_avgs(self) -> Dict[str, pd.DataFrame]
    def update_dec_avgs(self) -> Dict[str, pd.DataFrame]
    
    def update_all(self) -> Dict[str, pd.DataFrame]
```

**Design Notes**:
- Each method returns a new dictionary (immutability)
- Methods can be called individually for testing
- No side effects on input data

### 4. FeatureTransformer (Single Implementation)

**Responsibility**: Transform fighter stats into standardized feature format

**Interface**:
```python
class FeatureTransformer:
    def transform(
        self,
        fighter1_df: pd.DataFrame,
        fighter2_df: pd.DataFrame,
        available_stats: List[str]
    ) -> pd.DataFrame
```

**Output Format**:
Always generates three column types for each stat:
- `fighter1_<stat>`: Fighter1's absolute stat value
- `fighter2_<stat>`: Fighter2's absolute stat value
- `<stat>_diff`: Difference (fighter1 - fighter2)

**Key Design**:
- Single transformer implementation - no strategy pattern needed
- Always generates all three column types
- Feature filtering happens later in `predict.py` based on model requirements
- Simpler, more maintainable codebase

### 5. FeatureEnricher (Chain of Responsibility)

**Responsibility**: Add additional features (odds, experience, metadata)

**Interface**:
```python
class FeatureEnricher:
    def __init__(self, enrichers: List[Enricher])
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame
```

**Concrete Enrichers**:
- `OddsEnricher`: Adds BFO odds features
- `ExperienceEnricher`: Adds fight experience features
- `MetadataEnricher`: Adds metadata columns

**Design Notes**:
- Each enricher is independent
- Can be composed in any order
- Easy to add new enrichers

### 6. Feature Filtering (in predict.py)

**Responsibility**: Filter features based on model requirements before prediction

**Location**: `predict.py` (or a dedicated feature filtering utility)

**Interface**:
```python
def filter_features_for_model(
    df: pd.DataFrame,
    required_features: List[str]
) -> pd.DataFrame:
    """
    Filter DataFrame to only include columns that match required_features.
    
    Handles feature name patterns:
    - 'fighter1_<stat>' -> selects fighter1_<stat> column
    - 'fighter2_<stat>' -> selects fighter2_<stat> column
    - '<stat>_diff' -> selects <stat>_diff column
    - '<stat>' -> selects fighter1_<stat> column (for backward compatibility)
    """
    pass
```

**Key Design**:
- Feature filtering is separate from feature generation
- Each model's feature list determines what gets selected
- Win/loss models can use `fighter1_<stat>` and `<stat>_diff`
- Decision models can use `fighter1_<stat>`, `fighter2_<stat>`, and `<stat>_diff`
- No need for ModelType enum in inference data builder

## Data Flow

### Step-by-Step Pipeline

1. **Initialization**
   - Parse `fight_list` and `features`
   - Determine `ModelType` from features or explicit parameter
   - Initialize all components

2. **Data Loading**
   - `StaticDataLoader.load_fighter_data()` for each fighter
   - `DynamicDataLoader.load_fighter_data()` for each fighter
   - Filter out fights with insufficient data

3. **Stat Updates**
   - `StatUpdater.update_age()`
   - `StatUpdater.update_ufcage()`
   - `StatUpdater.update_days_since_last_fight()`
   - `StatUpdater.update_ratios()`
   - `StatUpdater.update_avgs()`
   - `StatUpdater.update_dec_avgs()`

4. **Data Combination**
   - Merge static and dynamic dataframes
   - Keep only final row (upcoming fight) for each fighter

5. **Feature Transformation**
   - Use single `FeatureTransformer` for all models
   - For each fight pair (fighter1, fighter2):
     - Generate `fighter1_<stat>`, `fighter2_<stat>`, and `<stat>_diff` for all available stats
     - Create output DataFrame with all three column types

6. **Feature Enrichment**
   - Apply `OddsEnricher`
   - Apply `ExperienceEnricher`
   - Apply `MetadataEnricher`

7. **Return Results**
   - Return dictionary: `{fighter1_name: DataFrame}`

## Feature Generation and Filtering

### Feature Generation (InferenceDataBuilder)

**Always generates** three column types for each stat:
- `fighter1_<stat>`: Fighter1's absolute stat value
- `fighter2_<stat>`: Fighter2's absolute stat value
- `<stat>_diff`: Difference (fighter1 - fighter2)

**Example**: For stat `time_sec_rd1_avg`, generates:
- `fighter1_time_sec_rd1_avg`
- `fighter2_time_sec_rd1_avg`
- `time_sec_rd1_avg_diff`

### Transformer Logic

```python
class FeatureTransformer:
    def transform(self, fighter1_df, fighter2_df, available_stats):
        """
        Generate all three column types for each available stat.
        """
        result = {}
        
        for stat in available_stats:
            # Skip metadata columns
            if stat in metadata_cols:
                continue
                
            # Generate fighter1_<stat>
            if stat in fighter1_df.columns:
                result[f'fighter1_{stat}'] = fighter1_df[stat]
            
            # Generate fighter2_<stat>
            if stat in fighter2_df.columns:
                result[f'fighter2_{stat}'] = fighter2_df[stat]
            
            # Generate <stat>_diff
            if stat in fighter1_df.columns and stat in fighter2_df.columns:
                result[f'{stat}_diff'] = fighter1_df[stat] - fighter2_df[stat]
        
        return pd.DataFrame(result)
```

### Feature Filtering (predict.py)

**Before model prediction**, filter DataFrame to only include required features:

```python
def filter_features_for_model(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame columns to match required_features list.
    
    Handles:
    - Exact matches: 'fighter1_time_sec_rd1_avg' -> selects that column
    - Backward compatibility: 'time_sec_rd1_avg' -> selects 'fighter1_time_sec_rd1_avg'
    """
    available_cols = set(df.columns)
    selected_cols = []
    
    for feat in required_features:
        if feat in available_cols:
            selected_cols.append(feat)
        elif feat.endswith('_diff'):
            # Already in correct format
            if feat in available_cols:
                selected_cols.append(feat)
        else:
            # Try fighter1_ prefix for backward compatibility
            fighter1_feat = f'fighter1_{feat}'
            if fighter1_feat in available_cols:
                selected_cols.append(fighter1_feat)
    
    return df[selected_cols]
```

### Model Feature Requirements

**Decision Model** (`DECISION_TEST_FEATS4`):
- Requires: `fighter1_<stat>`, `fighter2_<stat>`, `<stat>_diff`
- Example: `fighter1_time_sec_rd1_avg`, `fighter2_time_sec_rd1_avg`, `time_sec_rd1_avg_diff`

**Win/Loss Model**:
- Requires: `fighter1_<stat>` (or `<stat>`), `<stat>_diff`
- Example: `time_sec_rd1_avg` (maps to `fighter1_time_sec_rd1_avg`), `time_sec_rd1_avg_diff`

## Testing Strategy

### Unit Tests
- **DataLoader**: Test data loading and filtering
- **StatUpdater**: Test each update method independently
- **FeatureTransformer**: Test feature generation (all three column types)
- **FeatureEnricher**: Test each enricher independently
- **Feature Filtering**: Test filtering logic in predict.py

### Integration Tests
- **InferenceDataBuilder**: Test full pipeline with sample data
- **End-to-End**: Test with real CSV data

### Test Fixtures
- Mock CSV data
- Sample fight lists
- Expected output formats for each model type

## Migration Strategy

### Phase 1: Create New Architecture (Parallel)
- Implement new components alongside existing code
- Ensure backward compatibility

### Phase 2: Gradual Migration
- Update callers to use new `InferenceDataBuilder`
- Keep old `CreateInferenceData` as wrapper

### Phase 3: Deprecation
- Mark old class as deprecated
- Remove after all callers migrated

## Example Usage

### Inference Data Generation (Same for All Models)
```python
# Generate inference data with ALL features
builder = InferenceDataBuilder(
    csv_path="prediction_data.csv",
    fight_list=[("2024-01-01", "fighter1", "fighter2")],
    bfo_odds={"fighter1": -150}
)
fighter_dfs = builder.build()
# Result contains: fighter1_<stat>, fighter2_<stat>, <stat>_diff for ALL stats
```

### Win/Loss Model Prediction
```python
# In predict.py
fighter_df = fighter_dfs["fighter1"]
win_loss_features = WIN_LOSS_FEATS  # e.g., ['time_sec_rd1_avg', 'time_sec_rd1_avg_diff']

# Filter to only required features
filtered_df = filter_features_for_model(fighter_df, win_loss_features)
# filtered_df contains: fighter1_time_sec_rd1_avg, time_sec_rd1_avg_diff

# Make prediction
predictions = model.predict_proba(filtered_df)
```

### Decision Model Prediction
```python
# In predict.py
fighter_df = fighter_dfs["fighter1"]
decision_features = DECISION_TEST_FEATS4  # e.g., ['fighter1_time_sec_rd1_avg', 'fighter2_time_sec_rd1_avg', 'time_sec_rd1_avg_diff']

# Filter to only required features
filtered_df = filter_features_for_model(fighter_df, decision_features)
# filtered_df contains: fighter1_time_sec_rd1_avg, fighter2_time_sec_rd1_avg, time_sec_rd1_avg_diff

# Make prediction
predictions = model.predict_proba(filtered_df)
```

### Future: KO Model
```python
# Same inference data generation, just different feature filtering
ko_features = KO_MODEL_FEATS
filtered_df = filter_features_for_model(fighter_df, ko_features)
predictions = model.predict_proba(filtered_df)
```

## Benefits

1. **Separation of Concerns**: Each component has a single responsibility
2. **Simplicity**: Single transformer instead of multiple strategy implementations
3. **Testability**: Components can be tested in isolation
4. **Extensibility**: New model types only require feature list changes, no code changes
5. **Maintainability**: Clear structure makes code easier to understand
6. **Flexibility**: Can easily add new enrichers or feature types
7. **Efficiency**: Generate once, filter many times (if multiple models use same data)
8. **Backward Compatibility**: Feature filtering handles old feature names

## Implementation Plan

1. **Create Abstract Base Classes**
   - `DataLoader`, `FeatureTransformer`, `Enricher`

2. **Implement Concrete Classes**
   - `StaticDataLoader`, `DynamicDataLoader`
   - `StatUpdater`
   - `FeatureTransformer` (single implementation)
   - `OddsEnricher`, `ExperienceEnricher`

3. **Create Orchestrator**
   - `InferenceDataBuilder`

4. **Add Tests**
   - Unit tests for each component
   - Integration tests for full pipeline

5. **Add Feature Filtering**
   - Implement `filter_features_for_model()` in `predict.py` or utility module
   - Handle backward compatibility for old feature names

6. **Update Callers**
   - Update `predict.py` to use new builder
   - Add feature filtering before model prediction
   - Maintain backward compatibility

## File Structure

```
libs/feature_store/inference/
├── __init__.py
├── builder.py              # InferenceDataBuilder
├── loaders/
│   ├── __init__.py
│   ├── base.py            # DataLoader ABC
│   ├── static_loader.py
│   └── dynamic_loader.py
├── updaters/
│   ├── __init__.py
│   └── stat_updater.py
├── transformers/
│   ├── __init__.py
│   └── feature_transformer.py  # Single transformer implementation
└── enrichers/
    ├── __init__.py
    ├── base.py            # Enricher ABC
    ├── odds_enricher.py
    └── experience_enricher.py
```

## Conclusion

This refactored architecture provides:
- **Clear separation of concerns** with single-responsibility components
- **Simplicity** through single transformer that generates all feature types
- **High testability** through dependency injection and immutability
- **Easy extensibility** - new models only need feature list changes
- **Maintainability** through clear structure and interfaces
- **Flexibility** to support current and future model types
- **Efficiency** - generate once, filter as needed

The design follows Google engineering best practices: clean interfaces, dependency injection, separation of generation and filtering concerns, and comprehensive testing support.

**Key Insight**: By always generating all feature types (`fighter1_<stat>`, `fighter2_<stat>`, `<stat>_diff`) and filtering later, we eliminate the need for multiple transformers and make the system more maintainable and extensible.

