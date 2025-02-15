import optuna
import pandas as pd
from catboost import Pool, cv


class CatboostCVOptuna:
    def __init__(self, cat_feats: list, metric: str = "F:beta=2", fold_count: int = 3):
        self.cat_feats = cat_feats
        self.metric = metric
        self.fold_count = fold_count
        self.study = None

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna hyperparameter search.

        Defines hyperparameters as a dictionary, then performs cross-validation on the
        CatBoost model using the Pool and hyperparameters. Returns the best score
        obtained as a float.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object from Optuna.

        Returns
        -------
        score : float
            Best score obtained from cross-validation.
        """

        # Define hyperparameters
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=100),
            "depth": trial.suggest_int("depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 0.1, 10.0, log=True
            ),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "early_stopping_rounds": trial.suggest_int(
                "early_stopping_rounds", 20, 80, step=10
            ),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 0, 6),
            "loss_function": "Logloss",
            "eval_metric": self.metric,
            "custom_metric": ["Precision", "Recall"],
            "auto_class_weights": "Balanced",
            "verbose": False,
        }

        # Perform cross-validation
        cv_results = cv(
            self.pool,
            params,
            fold_count=self.fold_count,
            stratified=True,
            partition_random_seed=42,
            verbose_eval=False,
            plot=False,
        )

        return cv_results[f"test-{self.metric}-mean"].iloc[-1]  # Return best score

    def run(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 10):
        """
        Perform hyperparameter optimization using Optuna.

        :param X: Pandas DataFrame of feature data.
        :param y: Pandas Series of target data.
        :param n_trials: Number of hyperparameter optimization trials to run.
        :return: A dictionary of the best hyperparameters.
        """
        self.pool = Pool(data=X, label=y, cat_features=self.cat_feats)

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        self.params = self.study.best_params

        print("Best parameters:", self.params)
        print(f"Best {self.metric} score:", self.study.best_value)

        return self.get_best_params()

    def get_best_params(self):
        """
        Retrieve and update the best hyperparameters from the Optuna study.

        Modifies certain hyperparameters of the best trial,
        returns the dictionary of best hyperparameters.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters with updated fixed values.
        """

        self.params["loss_function"] = "Logloss"
        self.params["eval_metric"] = self.metric
        self.params["custom_metric"] = ["Precision", "Recall"]
        self.params["auto_class_weights"] = "Balanced"
        self.params["cat_features"] = self.cat_feats
        self.params["verbose"] = False
        return self.params
