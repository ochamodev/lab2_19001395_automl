import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def optimize_and_train(X, y, model_name, trials):
    global best_model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        if model_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 5, 20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model_name == "GradientBoosting":
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                               max_depth=max_depth, random_state=42)

        elif model_name == "SVM":
            C = trial.suggest_loguniform("C", 1e-3, 1e2)
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        elif model_name == "NaiveBayes":
            model = GaussianNB()
        else:
            raise ValueError(f"Modelo {model_name} no está soportado en la optimización.")

        # Entrenamiento y evaluación del modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, average='weighted')

    # Optimizamos los hiperparámetros usando Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    # Entrenamos el modelo con los mejores hiperparámetros
    best_params = study.best_params
    if model_name == "RandomForest":
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_name == "GradientBoosting":
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_name == "SVM":
        best_model = SVC(**best_params, probability=True, random_state=42)
    elif model_name == "KNN":
        best_model = KNeighborsClassifier(**best_params)
    elif model_name == "NaiveBayes":
        best_model = GaussianNB()

    best_model.fit(X, y)
    dump(best_model, "model/trained_model.joblib")
