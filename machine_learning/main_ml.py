import pandas as pd 
import ta
import optuna
import time
from multiprocessing import Pool
import plotly.graph_objects as go
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

random=1234

url = "https://raw.githubusercontent.com/Rerris/Technical_Analysis_Proyect/main/data/aapl_1d_test.csv"

def get_data(url):
    response = requests.get(url, verify=True)
    data = pd.read_csv(StringIO(response.text))
    data=data["Close"]
    data=pd.DataFrame(data)
    data["Closet-1"]=data["Close"].shift(-1)
    data["Closet-2"]=data["Close"].shift(-2)
    rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi_indicator.rsi()
    #MACD
    macd_indicator = ta.trend.MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()

    # Crear instancia del indicador EMA con períodos de 13 y 48
    ema_13_indicator = ta.trend.EMAIndicator(close=data['Close'], window=13)
    ema_48_indicator = ta.trend.EMAIndicator(close=data['Close'], window=48)
        
    # Calcular las EMAs 13 y 48 
    data['Ema 13'] = ema_13_indicator.ema_indicator()
    data['Ema 48'] = ema_48_indicator.ema_indicator()
    # Inicializar el indicador EMA con una ventana de 200 períodos
    ema_indicator = ta.trend.EMAIndicator(close=data['Close'], window=200)
    # Calcular la EMA 200

    data['Ema 200'] = ema_indicator.ema_indicator()
    data["Long"] = False
    data["Short"] = False

    for i in range(len(data["Close"])):

        if i + 10 < len(data["Close"]):
            if data["Close"][i] > data["Close"][i + 10]:
                data.loc[i, "Long"] = True
            elif data["Close"][i] < data["Close"][i + 10]:
                data.loc[i, "Short"] = True
    data=data
    return data

data=get_data(url).iloc[200:]

def logistic_regression_long(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Long']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo de regresión logística
    model = LogisticRegression()

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Imprimir el informe de clasificación
    #print(classification_report(y_test, y_pred))

logistic_regression_long(data)

def logistic_regression_short(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Short']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo de regresión logística
    model = LogisticRegression()

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Imprimir el informe de clasificación
    #print(classification_report(y_test, y_pred))

logistic_regression_short(data)

def support_vector_machine_long(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Long']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el clasificador SVM
    clf = svm.SVC(kernel='linear')  # Puedes cambiar el kernel según tus necesidades (lineal, polinomial, RBF, etc.)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

support_vector_machine_long(data)

def support_vector_machine_short(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Short']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el clasificador SVM
    clf = svm.SVC(kernel='linear')  # Puedes cambiar el kernel según tus necesidades (lineal, polinomial, RBF, etc.)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

support_vector_machine_short(data)

def XGBoost_long(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Long']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir los datos a un formato específico para XGBoost (DMatrix)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Definir los parámetros del modelo
    params = {
        'objective': 'binary:logistic',  # Problema de clasificación binaria
        'eval_metric': 'logloss',  # Métrica de evaluación
        'eta': 0.1,  # Tasa de aprendizaje
        'max_depth': 6,  # Profundidad máxima del árbol
        'subsample': 0.8,  # Proporción de muestras utilizadas para entrenar cada árbol
        'colsample_bytree': 0.8  # Proporción de características utilizadas para entrenar cada árbol
    }

    # Entrenar el modelo
    num_round = 100  # Número de iteraciones de entrenamiento (número de árboles)
    model = xgb.train(params, dtrain, num_round)

    # Predecir en el conjunto de prueba
    y_pred_proba = model.predict(dtest)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred_proba]  # Convertir probabilidades en etiquetas binarias

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

XGBoost_long(data)

def XGBoost_short(data):

    X = data[['RSI', 'Ema 13', 'Ema 200']]
    y = data['Short']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir los datos a un formato específico para XGBoost (DMatrix)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Definir los parámetros del modelo
    params = {
        'objective': 'binary:logistic',  # Problema de clasificación binaria
        'eval_metric': 'logloss',  # Métrica de evaluación
        'eta': 0.1,  # Tasa de aprendizaje
        'max_depth': 6,  # Profundidad máxima del árbol
        'subsample': 0.8,  # Proporción de muestras utilizadas para entrenar cada árbol
        'colsample_bytree': 0.8  # Proporción de características utilizadas para entrenar cada árbol
    }

    # Entrenar el modelo
    num_round = 100  # Número de iteraciones de entrenamiento (número de árboles)
    model = xgb.train(params, dtrain, num_round)

    # Predecir en el conjunto de prueba
    y_pred_proba = model.predict(dtest)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred_proba]  # Convertir probabilidades en etiquetas binarias

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

XGBoost_short(data)

data

# Optimizing using "Long" as the target variable
X = data[['RSI', 'Ema 13', 'Ema 200']]
y = data['Long']

# Optimizing Logistic Regression
def optimize_logistic_regression(trial):
    C = trial.suggest_loguniform('C', 0.01, 10) # Regularization parameter
    max_iter = trial.suggest_int('max_iter', 100, 1000) # Maximum number of iterations
    
    model = LogisticRegression(C=C, max_iter=max_iter)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Optimizing SVM
def optimize_svm(trial):
    C = trial.suggest_loguniform('C', 0.01, 10) # Regularization parameter
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']) # Kernel type
    
    model = svm.SVC(C=C, kernel=kernel)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Optimizing XGBoost
def optimize_xgboost(trial):
    eta = trial.suggest_loguniform('eta', 0.01, 0.1) # Learning rate
    max_depth = trial.suggest_int('max_depth', 3, 10) # Maximum depth of the tree
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0) # Subsample ratio of the training instances
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0) # Subsample ratio of columns when constructing each tree
    
    model = xgb.XGBClassifier(eta=eta, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Ojective function for Optuna
def objective(trial):
    model_name = trial.suggest_categorical('model', ['logistic_regression', 'svm', 'xgboost'])
    
    if model_name == 'logistic_regression':
        return optimize_logistic_regression(trial)
    elif model_name == 'svm':
        return optimize_svm(trial)
    elif model_name == 'xgboost':
        return optimize_xgboost(trial)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) # We decided to do 100 trials becasue we noticed that the the study convverges

# Get the best hyperparameters and corresponding score
best_params = study.best_trial.params
best_score = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)

# Print optimal hyperparameters
for param, value in best_params.items():
    print(f"Optimal {param}: {value}")


# Optimizing using "Short" as the target variable
X = data[['RSI', 'Ema 13', 'Ema 200']]
y = data['Short']

# Optimizing Logistic Regression
def optimize_logistic_regression(trial):
    C = trial.suggest_loguniform('C', 0.01, 10) # Regularization parameter
    max_iter = trial.suggest_int('max_iter', 100, 1000) # Maximum number of iterations
    
    model = LogisticRegression(C=C, max_iter=max_iter)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Optimizing SVM
def optimize_svm(trial):
    C = trial.suggest_loguniform('C', 0.01, 10) # Regularization parameter
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']) # Kernel type
    
    model = svm.SVC(C=C, kernel=kernel)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Optimizing XGBoost
def optimize_xgboost(trial):
    eta = trial.suggest_loguniform('eta', 0.01, 0.1) # Learning rate
    max_depth = trial.suggest_int('max_depth', 3, 10) # Maximum depth of the tree
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0) # Subsample ratio of the training instances
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0) # Subsample ratio of columns when constructing each tree
    
    model = xgb.XGBClassifier(eta=eta, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
    scores = cross_val_score(model, X, y, cv=5) # 5-fold cross-validation
    
    return scores.mean()

# Ojective function for Optuna
def objective(trial):
    model_name = trial.suggest_categorical('model', ['logistic_regression', 'svm', 'xgboost'])
    
    if model_name == 'logistic_regression':
        return optimize_logistic_regression(trial)
    elif model_name == 'svm':
        return optimize_svm(trial)
    elif model_name == 'xgboost':
        return optimize_xgboost(trial)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) # We decided to do 100 trials becasue we noticed that the the study converges

# Get the best hyperparameters and corresponding score
best_params = study.best_trial.params
best_score = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)

# Print optimal hyperparameters
for param, value in best_params.items():
    print(f"Optimal {param}: {value}")

# Backtesting functions
def backtest_logistic_regression(data, model, target, take_profit, stop_loss, initial_capital, margin_rate):
    # Crear una copia de los datos
    data = data.copy()
    
    # Crear una nueva columna con las predicciones del modelo
    data['Prediction'] = model.predict(data[['RSI', 'Ema 13', 'Ema 200']])
    
    # Calcular el rendimiento diario
    data['Return'] = data['Closet-1'] / data['Close'] - 1
    
    # Calcular el rendimiento diario de la estrategia
    data['Strategy Return'] = data['Return'] * data['Prediction']
    
    # Calcular el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'] = (data['Strategy Return'] + 1).cumprod()
    
    # Inicializar el capital
    capital = initial_capital
    
    # Inicializar el estado de la operación
    in_trade = False
    
    # Inicializar el precio de entrada
    entry_price = 0
    
    # Recorrer los datos
    for i in range(len(data)):
        # Si estamos en una operación
        if in_trade:
            # Si alcanzamos el take profit o el stop loss
            if data['Close'].iloc[i] >= entry_price * (1 + take_profit) or data['Close'].iloc[i] <= entry_price * (1 - stop_loss):
                # Salir de la operación
                in_trade = False
                # Agregar los fondos de margen utilizados al capital
                capital += (entry_price - data['Close'].iloc[i]) * margin_rate
        # Si no estamos en una operación y el modelo predice una entrada
        elif data['Prediction'].iloc[i] == 1:
            # Entrar en la operación
            in_trade = True
            entry_price = data['Close'].iloc[i]
            capital -= entry_price  # Descontar el precio de entrada del capital
    
    # Graficar el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'].plot(figsize=(10, 6))
    plt.title(f'Cumulative Strategy Return ({target})')
    plt.show()
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(data[target], data['Prediction'])
    print("Accuracy:", accuracy)
    
    # Imprimir el informe de clasificación
    print(classification_report(data[target], data['Prediction']))
    
    # Imprimir la matriz de confusión
    print("Confusion Matrix:")
    print(confusion_matrix(data[target], data['Prediction']))
    
    # Imprimir el capital final
    print("Final capital:", capital)

# Probar la estrategia con el modelo de regresión logística para "Long"
model = LogisticRegression(C=0.01, max_iter=100)
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Long'])
backtest_logistic_regression(data, model, 'Long', 0.05, 0.02, 100000)

# Probar la estrategia con el modelo de regresión logística para "Short"
model = LogisticRegression(C=0.01, max_iter=100)
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Short'])
backtest_logistic_regression(data, model, 'Short', 0.05, 0.02, 100000)

#Funcion de backtest para SVM
def backtest_svm(data, model, target, take_profit, stop_loss, initial_capital, margin_rate):
    # Crear una copia de los datos
    data = data.copy()
    
    # Crear una nueva columna con las predicciones del modelo
    data['Prediction'] = model.predict(data[['RSI', 'Ema 13', 'Ema 200']])
    
    # Calcular el rendimiento diario
    data['Return'] = data['Closet-1'] / data['Close'] - 1
    
    # Calcular el rendimiento diario de la estrategia
    data['Strategy Return'] = data['Return'] * data['Prediction']
    
    # Calcular el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'] = (data['Strategy Return'] + 1).cumprod()
    
    # Inicializar el capital
    capital = initial_capital
    
    # Inicializar el estado de la operación
    in_trade = False
    
    # Inicializar el precio de entrada
    entry_price = 0
    
    # Recorrer los datos
    for i in range(len(data)):
        # Si estamos en una operación
        if in_trade:
            # Si alcanzamos el take profit o el stop loss
            if data['Close'].iloc[i] >= entry_price * (1 + take_profit) or data['Close'].iloc[i] <= entry_price * (1 - stop_loss):
                # Salir de la operación
                in_trade = False
                # Agregar los fondos de margen utilizados al capital
                capital += (entry_price - data['Close'].iloc[i]) * margin_rate
        # Si no estamos en una operación y el modelo predice una entrada
        elif data['Prediction'].iloc[i] == 1:
            # Entrar en la operación
            in_trade = True
            entry_price = data['Close'].iloc[i]
            capital -= entry_price  # Descontar el precio de entrada del capital
    
    # Graficar el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'].plot(figsize=(10, 6))
    plt.title(f'Cumulative Strategy Return ({target})')
    plt.show()
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(data[target], data['Prediction'])
    print("Accuracy:", accuracy)
    
    # Imprimir el informe de clasificación
    print(classification_report(data[target], data['Prediction']))
    
    # Imprimir la matriz de confusión
    print("Confusion Matrix:")

    # Imprimir el capital final
    print("Final capital:", capital)

# Probar la estrategia con el modelo SVM para "Long"
model = svm.SVC(C=0.01, kernel='rbf')
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Long'])
backtest_svm(data, model, 'Long', 0.05, 0.02, 100000)

# Probar la estrategia con el modelo SVM para "Short"
model = svm.SVC(C=0.01, kernel='rbf')
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Short'])
backtest_svm(data, model, 'Short', 0.05, 0.02, 100000)

# Funcion de backtest para XGBoost
def backtest_xgboost(data, model, target, take_profit, stop_loss, initial_capital, margin_rate):
    # Crear una copia de los datos
    data = data.copy()
    
    # Crear una nueva columna con las predicciones del modelo
    data['Prediction'] = model.predict(data[['RSI', 'Ema 13', 'Ema 200']])
    
    # Calcular el rendimiento diario
    data['Return'] = data['Closet-1'] / data['Close'] - 1
    
    # Calcular el rendimiento diario de la estrategia
    data['Strategy Return'] = data['Return'] * data['Prediction']
    
    # Calcular el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'] = (data['Strategy Return'] + 1).cumprod()
    
    # Inicializar el capital
    capital = initial_capital
    
    # Inicializar el estado de la operación
    in_trade = False
    
    # Inicializar el precio de entrada
    entry_price = 0
    
    # Recorrer los datos
    for i in range(len(data)):
        # Si estamos en una operación
        if in_trade:
            # Si alcanzamos el take profit o el stop loss
            if data['Close'].iloc[i] >= entry_price * (1 + take_profit) or data['Close'].iloc[i] <= entry_price * (1 - stop_loss):
                # Salir de la operación
                in_trade = False
                # Agregar los fondos de margen utilizados al capital
                capital += (entry_price - data['Close'].iloc[i]) * margin_rate
        # Si no estamos en una operación y el modelo predice una entrada
        elif data['Prediction'].iloc[i] == 1:
            # Entrar en la operación
            in_trade = True
            entry_price = data['Close'].iloc[i]
            capital -= entry_price  # Descontar el precio de entrada del capital
    
    # Graficar el rendimiento acumulado de la estrategia
    data['Cumulative Strategy Return'].plot(figsize=(10, 6))
    plt.title(f'Cumulative Strategy Return ({target})')
    plt.show()
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(data[target], data['Prediction'])
    print("Accuracy:", accuracy)
    
    # Imprimir el informe de clasificación
    print(classification_report(data[target], data['Prediction']))
    
    # Imprimir la matriz de confusión
    print("Confusion Matrix:")

    # Imprimir el capital final
    print("Final capital:", capital)

# Probar la estrategia con el modelo XGBoost para "Long"
model = xgb.XGBClassifier(eta=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8)
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Long'])
backtest_xgboost(data, model, 'Long', 0.05, 0.02, 100000)

# Probar la estrategia con el modelo XGBoost para "Short"
model = xgb.XGBClassifier(eta=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8)
model.fit(data[['RSI', 'Ema 13', 'Ema 200']], data['Short'])
backtest_xgboost(data, model, 'Short', 0.05, 0.02, 100000)