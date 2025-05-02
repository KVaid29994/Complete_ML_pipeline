# Add stage: data_ingestion
dvc stage add -n data_ingestion `
  -d src/data_ingestion.py `
  -p data_ingestion.test_size `
  -o data/raw `
  python src/data_ingestion.py

# Add stage: preprocessing
dvc stage add -n preprocessing `
  -d data/raw `
  -d src/preprocessing.py `
  -o data/interim `
  python src/preprocessing.py

# Add stage: feature_engineering
dvc stage add -n feature_engineering `
  -d data/interim `
  -d src/feature_engineering.py `
  -p feature_engineering.max_features `
  -o data/processed `
  python src/feature_engineering.py

# Add stage: model_building
dvc stage add -n model_training `
  -d data/processed `
  -d src/model_training.py `
  -p model_training.n_estimators `
  -p model_training.random_state `
  -o models/random_forest_model.pkl `
  python src/model_training.py

# Add stage: model_evaluation
dvc stage add -n model_evaluation `
  -d models/random_forest_model.pkl `
  -d src/model_evaluation.py `
  -M reports/metrics.json `
  python src/model_evaluation.py
