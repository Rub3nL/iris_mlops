import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


from iris_mlops.config import(
    FEATURES,
    METRICS_PATH,
    MIN_ACCURACY,
    MODELS_PARAMS,
    MODEL_PATH,
)

from iris_mlops.data import load_data, split_data