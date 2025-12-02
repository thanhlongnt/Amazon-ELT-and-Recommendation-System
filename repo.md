1. Predictive task
- Task: Predict a user’s next purchase category at time t+1 given their historical prefix up to time t.
- Evaluation protocol:
  - User-level split via shard-based data: train/val/test by sharded files in sequence_samples_by_shard produced by `common_scripts.04_create_train_data.main`.
  - Metrics: accuracy and top-3 accuracy reported in modeling scripts:
    - `modeling.train_logreg_next_category.main`,
    - `modeling.train_histgbm_next_category.main`,
    - `modeling.train_other_models_next_category.main`.
- Baselines:
  - Global majority class (from train),
  - Last-category baseline (`last_category_idx`),
  - Prefix-most-frequent category baseline (`prefix_most_freq_category_idx`).
  - Implemented in each modeling script (see baselines blocks in the three files above) and stream-computed in `common_scripts.BaselineStats`.
- Validity checks:
  - Shard-level splits prevent user leakage.
  - Classification report per class distribution (e.g., printed in `modeling.train_histgbm_next_category.evaluate_histgbm`).
  - Compare against trivial baselines to ensure non-trivial improvement.

2. Exploratory analysis, data collection, pre-processing, and discussion
- Context:
  - Source: Amazon Reviews 2023 (UCSD/McAuley Lab). Raw per-category JSONL.GZ files and item meta. Download and sync managed via the data registry and Google Drive in `common_scripts.data_io`.
  - Purpose: Build a sequential dataset to study user category transitions.
- Processing steps:
  - Script 03 filters reviews to “top users” and produces per-review and aggregated features via `common_scripts.03_user_features.process_category`, streaming parse helpers like `common_scripts.parse_reviews_for_top_users`.
  - Script 04 shards both reviews and user features by user_id, then builds temporal sequence samples:
    - Sharding: `common_scripts.shard_user_features` and `common_scripts.shard_reviews_by_user`.
    - Sequence building per shard: `common_scripts.build_sequence_dataset_for_shard` constructs prefix features, last-N category indices, prefix category counts, and labels the next category.
    - Streaming baselines: `common_scripts.BaselineStats`.
  - Outputs:
    - Per-shard sequence files in sequence_samples_by_shard,
    - Global merged dataset: sequence_training_samples.parquet.

3. Modeling
- Formulation:
  - Inputs: Prefix-derived features at time t (static user features, prefix stats, last ratings/helpful/item avg, last-N category indices, prefix category counts, prefix-most-frequent category).
  - Output: Target category index at time t+1 (`target_category_idx`).
  - Objective: Multiclass classification, optimizing log-loss or cross-entropy (e.g., logistic regression; HistGBM uses log_loss).
- Models implemented (course-relevant + stronger baselines):
  - Multinomial Logistic Regression: `modeling.train_logreg_next_category.main` with preprocessing pipelines and derived features via `modeling.add_derived_features`.
  - Decision Tree / Random Forest: `modeling.train_other_models_next_category.main`.
  - HistGradientBoostingClassifier: `modeling.train_histgbm_next_category.main`.
- Discussion:
  - Logistic Regression: scalable, interpretable with OHE for categorical idx features; may underfit complex dynamics.
  - Decision Tree/Random Forest: handle mixed types without OHE; stronger nonlinearity; more memory and slower inference.
  - HistGBM: efficient, strong performance with early stopping; needs dense numeric arrays and careful NaN handling.
- Code architecture:
  - Split discovery and loading: `list_shard_files`/`load_split_from_shards` in both histgbm/other models.
  - Preprocessing:
    - OHE + scaling for linear models (`ColumnTransformer`), see `modeling.train_logreg_next_category` and `modeling.train_other_models_next_category`.
    - Median imputation and dense arrays for HistGBM via `modeling.prepare_features_for_histgbm`.
  - Evaluation helpers: per-split accuracy + top-3, classification report for best model.

4. Evaluation
- Metrics:
  - Accuracy: primary metric for multiclass next-category prediction.
  - Top-3 accuracy: measures whether the true next category is within the top-3 predicted classes, appropriate when there are many classes and user choice uncertainty.
  - Classification report: per-class precision/recall/f1 to assess class imbalance and behavior.
- Baselines:
  - Implemented and printed in each modeling script; provide anchors:
    - Global majority accuracy (e.g., ~0.115 on val),
    - Last-category and prefix-most-frequent (e.g., ~0.235–0.241 on val).
- Demonstrating improvements:
  - Random Forest/Decision Tree and Logistic Regression beat trivial baselines; HistGBM typically best (see saved logs like histgbm_results.txt, random_forest.txt, decision_tree.txt, data/train_logres.txt).
- Implementation:
  - Baselines computation blocks in:
    - `modeling.train_logreg_next_category.main`,
    - `modeling.train_histgbm_next_category.main`,
    - `modeling.train_other_models_next_category.main`.
  - Evaluation functions:
    - `modeling.train_histgbm_next_category.evaluate_histgbm`,
    - `modeling.train_other_models_next_category.evaluate_model`,
    - Inline eval in `modeling.train_logreg_next_category.main`.

5. Related work
- Dataset usage:
  - Amazon Reviews data is widely used for recommendation, next-item/category prediction, and user behavior modeling.
- Prior approaches:
  - Sequential recommendation models (Markov chains, factorization, GRU4Rec/Seq2Seq/Transformers), tree-based and linear classifiers for simpler feature-based formulations.
- Comparison:
  - Your feature-based sequential classification aligns with classic baselines and efficient, scalable learners (LogReg, RF, GBM). Results in the logs show consistent improvements over trivial baselines; deeper neural sequence models could further improve at higher complexity cost, but the current models are appropriate and directly relevant to course content.

Key code entry points to cite in your report:
- Data prep and sequence building:
  - `common_scripts.04_create_train_data.main`,
  - `common_scripts.build_sequence_dataset_for_shard`,
  - `common_scripts.BaselineStats`.
- Modeling:
  - `modeling.train_logreg_next_category.main`,
  - `modeling.train_histgbm_next_category.main`,
  - `modeling.train_other_models_next_category.main`.