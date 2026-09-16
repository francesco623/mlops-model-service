from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.xgboost
import shutil
import optuna
import pandas as pd
import xgboost as xgb

data = pd.read_csv("dataset/data.csv")
target = pd.read_csv("dataset/target.csv")

X = data.select_dtypes(include='number')  # keep numeric columns only
X['sex'] = (data['sex'] == 'male').astype(int)  # male=1, female=0, keeping only numbers exclude the column "sex", so we convert it to number

y = target['survived'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Xgboost needs Dmatrix
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

# Create mlflow db and set experiment name
mlflow.set_tracking_uri("sqlite:///mlflowP2.db")
# use: mlflow ui --backend-store-uri sqlite:///mlflowP2.db 

mlflow.set_experiment("titanic-P2")

MODEL_DIR = "model"

FIXED_PARAMS = {
    "objective": "binary:logistic",
    "seed": 42,
}

def train_booster(params):
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,        # maximum rounds
        evals=[(dtest, "validation")],
        early_stopping_rounds=50,    # stops after 50 rounds without improvement
        verbose_eval=False,
    )

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    } | FIXED_PARAMS

    # num_boost_round = trial.suggest_int("num_boost_round", 100, 500) # -> if you wanna use optuna to optimize this param too, make sure to record it


    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        booster = train_booster(params)

        preds = booster.predict(dtest) # -> make prediction on the test dataset, note: this prediction is a probability [0.34, 0.56, 0.47...]
        preds_binary = (preds > 0.5).astype(int) # -> we convert the prediction set to true or false, based on > 0.5 condition, and then to 1 or 0 with .astype(int)
        acc = accuracy_score(y_test, preds_binary) # -> we compare our model prediction with the real answer and store it in acc as a score
        mlflow.log_metric("accuracy", acc) # -> let mlflow record our score

    return acc # -> we pass our score to optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best acc: {study.best_value}")

# Final model: same training as the trials, with the best params
final_booster = train_booster(study.best_params | FIXED_PARAMS)

final_acc = accuracy_score(y_test, (final_booster.predict(dtest) > 0.5).astype(int))
print(f"Final model acc: {final_acc}")  # must match Best acc

shutil.rmtree(MODEL_DIR, ignore_errors=True)  # save_model fails if the folder exists
mlflow.xgboost.save_model(final_booster, MODEL_DIR)