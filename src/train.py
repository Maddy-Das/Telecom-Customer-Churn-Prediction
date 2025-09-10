import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from src.preprocess import preprocess_data
from src.utils import load_data, evaluate_model, save_model

def train_models():
    df = load_data()
    X, y, scaler, _, _ = preprocess_data(df, use_smote=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'XGBoost':
            grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best params for {name}: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics, cm = evaluate_model(y_test, y_pred)
        results[name] = {'model': model, 'metrics': metrics, 'confusion_matrix': cm}
        print(f"{name} Metrics:", metrics)
        print(f"{name} Confusion Matrix:\n{cm}\n")

    for name, result in results.items():
        save_model(result['model'], f"{name.lower().replace(' ', '_')}_model.pkl")
    save_model(scaler, 'scaler.pkl')
    print("Models saved.")

    return X_test, y_test, results, scaler

if __name__ == "__main__":
    train_models()