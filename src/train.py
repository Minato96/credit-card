import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

from preprocess import load_data, scale_features, get_features_and_target, apply_smote
from evaluate import print_classification_report, plot_and_save_confusion_matrix

def main():
    df = load_data('../data/creditcard.csv')
    df_scaled, scaler = scale_features(df)
    X, y = get_features_and_target(df_scaled, 'Class')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

    X_train_resampled,Y_train_resampled = apply_smote(X_train, y_train)

    models = {
        'logitic' : LogisticRegression(max_iter=1000),
        'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost' : XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state = 42)
    }
    best_model = None
    best_recall = 0

    for name,model in models.items():
        print(f"----Training {name}----")
        model.fit(X_train_resampled,Y_train_resampled)
        y_pred = model.predict(X_test)

        print_classification_report(y_test,y_pred)

        report = classification_report(y_test, y_pred,target_names=['No Fraud', 'Fraud'], output_dict=True)
        recall_fraud = report['Fraud']['recall']

        if recall_fraud > best_recall:
            best_recall = recall_fraud
            best_model = model
        
    print(f"\nBest model is {type(best_model).__name__} with fraud recall of {best_recall:.2f}")
    plot_and_save_confusion_matrix(best_model, X_test, y_test, '../results/confusion_matrix.png')

    joblib.dump(best_model, 'best_fraud_detector.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Best model and scaler have been saved.")

if __name__ == '__main__':
    main()