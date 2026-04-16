import optuna
from sklearn.model_selection import cross_val_score

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.logging import logger

from xgboost import XGBClassifier

# params suggestions
def suggest_params(trial, param_config):
    params = {}

    for param, details in param_config.items():
        if details["type"] == "int":
            params[param] = trial.suggest_int(param, details["low"], details["high"])

        elif details["type"] == "float":
            params[param] = trial.suggest_float(param, details["low"], details["high"])

        elif details["type"] == "categorical":
            params[param] = trial.suggest_categorical(param, details["choices"])
    logger.debug(f"Suggested params: {params}")
    return params


# model factory
def get_model(model_type, params):
    logger.info(f"Initializing model: {model_type}")

    if model_type == "xgboost":
        return XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric="mlogloss")

    elif model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=42)

    elif model_type == "logistic_regression":
        return LogisticRegression(**params)

    elif model_type == "svm":
        return SVC(**params)

    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")


# objective function
def create_objective(X, y, config):
    def objective(trial):
        model_type = trial.suggest_categorical(
            "model", config.models
        )

        param_config = config.params.get(model_type, {})
        params = suggest_params(trial, param_config)

        model = get_model(model_type, params)

        score = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        logger.info(f"Trial {trial.number} completed | Model: {model_type} | Score: {score:.4f}")

        return score

    return objective

# tuning function
def tune_model(X, y, config):
    logger.info("Starting Optuna hyperparameter tuning...")

    study = optuna.create_study(
        direction=config.direction
    )

    objective = create_objective(X, y, config)

    study.optimize(objective, n_trials=config.n_trials)

    best_model_type = study.best_params["model"]

    # Remove model key from params
    best_params = {k: v for k, v in study.best_params.items() if k != "model"}

    logger.info(f"Best model found: {best_model_type}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {study.best_value:.4f}")

    best_model = get_model(best_model_type, best_params)

    logger.info("Training best model on full dataset")
    best_model.fit(X, y)

    logger.info("Model training completed")

    return {
        "model": best_model,
        "model_type": best_model_type,
        "best_params": best_params,
        "best_score": study.best_value
    }