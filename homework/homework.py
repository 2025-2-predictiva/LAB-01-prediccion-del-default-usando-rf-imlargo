# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_and_clean_data():
    """
    Paso 1: Load and clean datasets
    - Rename 'default payment next month' to 'default'
    - Remove 'ID' column
    - Remove records with missing information
    - Group EDUCATION values > 4 into 'others' category (value 4)
    """
    # Load datasets
    train_df = pd.read_csv("files/input/train_default_of_credit_card_clients.csv")
    test_df = pd.read_csv("files/input/test_default_of_credit_card_clients.csv")

    # Rename target column
    train_df = train_df.rename(columns={"default payment next month": "default"})
    test_df = test_df.rename(columns={"default payment next month": "default"})

    # Remove ID column
    train_df = train_df.drop(columns=["ID"])
    test_df = test_df.drop(columns=["ID"])

    # Remove records with missing information (0 values in EDUCATION and MARRIAGE)
    train_df = train_df[train_df["EDUCATION"] != 0]
    train_df = train_df[train_df["MARRIAGE"] != 0]
    test_df = test_df[test_df["EDUCATION"] != 0]
    test_df = test_df[test_df["MARRIAGE"] != 0]

    # Group EDUCATION values > 4 into category 4 (others)
    train_df.loc[train_df["EDUCATION"] > 4, "EDUCATION"] = 4
    test_df.loc[test_df["EDUCATION"] > 4, "EDUCATION"] = 4

    return train_df, test_df


def split_data(train_df, test_df):
    """
    Paso 2: Split datasets into X and y
    """
    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """
    Paso 3: Create pipeline with OneHotEncoder and RandomForestClassifier
    """
    # Identify categorical columns
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    # Create column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough",
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimize hyperparameters using GridSearchCV with 10-fold CV
    """
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def save_model(model):
    """
    Paso 5: Save model compressed with gzip
    """
    # Create models directory if it doesn't exist
    os.makedirs("files/models", exist_ok=True)

    # Save model compressed with gzip
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6 y 7: Calculate and save metrics and confusion matrices
    """
    # Create output directory if it doesn't exist
    os.makedirs("files/output", exist_ok=True)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics for train set
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred),
    }

    # Calculate metrics for test set
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred),
    }

    # Calculate confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Format confusion matrix for train
    train_cm = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
    }

    # Format confusion matrix for test
    test_cm = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
    }

    # Save metrics to file (one JSON object per line)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")
        f.write(json.dumps(train_cm) + "\n")
        f.write(json.dumps(test_cm) + "\n")


def main():
    """
    Main function to execute all steps
    """
    # Paso 1: Load and clean data
    print("Step 1: Loading and cleaning data...")
    train_df, test_df = load_and_clean_data()

    # Paso 2: Split data
    print("Step 2: Splitting data...")
    x_train, y_train, x_test, y_test = split_data(train_df, test_df)

    # Paso 3: Create pipeline
    print("Step 3: Creating pipeline...")
    pipeline = create_pipeline(x_train)

    # Paso 4: Optimize hyperparameters
    print("Step 4: Optimizing hyperparameters...")
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Paso 5: Save model
    print("Step 5: Saving model...")
    save_model(model)

    # Paso 6 y 7: Calculate and save metrics
    print("Step 6 & 7: Calculating and saving metrics...")
    calculate_and_save_metrics(model, x_train, y_train, x_test, y_test)

    print("Done! Model and metrics saved successfully.")


if __name__ == "__main__":
    main()
