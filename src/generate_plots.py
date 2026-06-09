from preprocess import preprocess_pipeline
from evaluate import load_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance

DATA_PATH = "../data/BankChurners.csv"
MODEL_PATH = "../models/best_model.pkl"

X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH, sampling="smote")

model = load_model(MODEL_PATH)
y_pred = model.predict(X_test)

plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(model, X_test, y_test)
plot_feature_importance(model, X_test.columns)