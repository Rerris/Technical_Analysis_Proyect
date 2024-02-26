
# # Project 1: Technical Analysis: EMA 13/48 Crossover 

# %%
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
from io import StringIO
import numpy as np
import optuna

# %%
url = "https://raw.githubusercontent.com/Rerris/Technical_Analysis_Proyect/main/aapl_1s_data%20(1).csv"

response = requests.get(url, verify=True)
data = pd.read_csv(StringIO(response.text))

# %%
#RSI
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

# %%
def rsi_long(data, window):
    rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window= window)
    data['RSI'] = rsi_indicator.rsi()
    data["RSI_TF_Long"]=False
    
    for i in range(len(data)):
        # Obtener el valor actual de RSI
        rsi_value = data['RSI'].iloc[i]

        # Si el valor de RSI es mayor que 50, establecer "True" en la columna "RSI_TF"
        if rsi_value > 50:
            data["RSI_TF_Long"].iloc[i] = True

    return data
            

# %%
window = 14

# %%
rsi_long(data,window)

# %%
def rsi_short(data, window):
    rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=window)
    data['RSI'] = rsi_indicator.rsi()
    data["RSI_TF_Short"]=False
    
    for i in range(len(data)):
        # Obtener el valor actual de RSI
        rsi_value = data['RSI'].iloc[i]

        # Si el valor de RSI es mayor que 50, establecer "True" en la columna "RSI_TF"
        if rsi_value > 50:
            data["RSI_TF_Short"].iloc[i] = True

    return data

# %%
window = 14

# %%
rsi_short(data,window)

# %%
def macd_long(data, window_slow, window_fast, window_sign):
    macd_indicator = ta.trend.MACD(close=data['Close'], window_slow = window_sign, window_fast = window_fast, window_sign = window_sign)
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()

    # Últimos valores de MACD y señal
    macd = data['macd']
    macd_signal = data['macd_signal']

    data["MACD_TF"] = False

    for i in range(1, len(data)):
        # Si el MACD cruza hacia arriba la línea de señal en este punto de datos
        if macd.iloc[i] > macd_signal.iloc[i] and macd.iloc[i - 1] <= macd_signal.iloc[i - 1]:
            data["MACD_TF"].iloc[i] = True

    return data

# %%
window_slow = 26
window_fast = 12
window_sign = 9

# %%
macd_long(data, window_slow, window_fast, window_sign)

# %%
def macd_short(data, window_slow, window_fast, window_sign):
    macd_indicator = ta.trend.MACD(close=data['Close'], window_slow = window_sign, window_fast = window_fast, window_sign = window_sign)
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()

    # Últimos valores de MACD y señal
    macd = data['macd']
    macd_signal = data['macd_signal']

    data["MACD_TF_Short"] = False

    for i in range(1, len(data)):
        # Si el MACD cruza hacia arriba la línea de señal en este punto de datos
        if macd.iloc[i] < macd_signal.iloc[i] and macd.iloc[i - 1] <= macd_signal.iloc[i - 1]:
            data["MACD_TF_Short"].iloc[i] = True

    return data

# %%
macd_short(data, window_slow, window_fast, window_sign)

# %%
def ema_cross_long(data, ema_13_window, ema_48_window):
    # Crear instancia del indicador EMA con períodos de 13 y 48
    ema_13_indicator = ta.trend.EMAIndicator(close=data['Close'], window=ema_13_window)
    ema_48_indicator = ta.trend.EMAIndicator(close=data['Close'], window=ema_48_window)
    
    # Calcular las EMAs
    ema_13 = ema_13_indicator.ema_indicator()
    ema_48 = ema_48_indicator.ema_indicator()

    # Inicializar la señal de cruce a False
    data["EMA_cross_TF_Long"] = False

    # Iterar sobre los índices desde el segundo valor
    for i in range(1, len(data)):
        # Si hay un cruce alcista de la EMA de 13 sobre la EMA de 48
        """
        El siguiente codigo es por si se quiere hacer con con la compra al hacer el cruce unicamente cuando sucede
        if data['Ema 13'].iloc[i] > data['Ema 48'].iloc[i] and data['Ema 13'].iloc[i - 1] <= data['Ema 48'].iloc[i - 1] and not data["EMA_cross_TF_Long"].iloc[i - 1]:

        """
        if data['Ema 13'].iloc[i] > data['Ema 48'].iloc[i] and data['Ema 13'].iloc[i - 1] <= data['Ema 48'].iloc[i - 1]:
            data["EMA_cross_TF_Long"].iloc[i] = True

    return data

# %%
ema_13_window = 13
ema_48_window = 48

# %%
ema_cross_long(data, ema_13_window, ema_48_window)

# %%
def ema_cross_short(data, ema_13_window, ema_48_window):
    # Crear instancia del indicador EMA con períodos de 13 y 48
    ema_13_indicator = ta.trend.EMAIndicator(close=data['Close'], window=ema_13_window)
    ema_48_indicator = ta.trend.EMAIndicator(close=data['Close'], window=ema_48_window)
    
    # Calcular las EMAs
    ema_13 = ema_13_indicator.ema_indicator()
    ema_48 = ema_48_indicator.ema_indicator()

    # Inicializar la señal de cruce a False
    data["EMA_cross_TF_Short"] = False

    # Iterar sobre los índices desde el segundo valor
    for i in range(1, len(data)):
        """
        El siguiente codigo es por si se quiere hacer con con la compra al hacer el cruce unicamente cuando sucede
        
         if data['Ema 13'].iloc[i] < data['Ema 48'].iloc[i] and data['Ema 13'].iloc[i - 1] >= data['Ema 48'].iloc[i - 1] and not data["EMA_cross_TF_Short"].iloc[i - 1]:
        """
        # Si hay un cruce alcista de la EMA de 13 sobre la EMA de 48
        if data['Ema 13'].iloc[i] > data['Ema 48'].iloc[i] and data['Ema 13'].iloc[i - 1] <= data['Ema 48'].iloc[i - 1]:
            data["EMA_cross_TF_Short"].iloc[i] = True

    return data

# %%
ema_cross_short(data, ema_13_window, ema_48_window)

# %%
def ema_200_long(data, ema_200_window):
    # Inicializar el indicador EMA con una ventana de 200 períodos
    ema_indicator = ta.trend.EMAIndicator(close=data['Close'], window = ema_200_window)
    
    # Calcular la EMA
    ema_200 = ema_indicator.ema_indicator()
    
    # Inicializar la columna de señal de compra
    data["Ema 200 TF Long"] = False

    # Iterar sobre los índices desde el primer valor
    for i in range(len(data)):
        # Obtener el valor actual de la EMA de 200
        ema_200_value = data["Ema 200"].iloc[i]

        # Obtener el último precio de cierre
        ultimo_precio_cierre = data['Close'].iloc[i]

        # Si el precio está por encima de la EMA de 200, establecer la señal de compra en True
        if ultimo_precio_cierre > ema_200_value:
            data.at[i, "Ema 200 TF Long"] = True

    return data

# %%
def ema_200_short(data, ema_200_window):
    # Inicializar el indicador EMA con una ventana de 200 períodos
    ema_indicator = ta.trend.EMAIndicator(close=data['Close'], window=ema_200_window)
    
    # Calcular la EMA
    ema_200 = ema_indicator.ema_indicator()
    
    # Inicializar la columna de señal de compra
    data["Ema 200 TF Short"] = False

    # Iterar sobre los índices desde el primer valor
    for i in range(len(data)):
        # Obtener el valor actual de la EMA de 200
        ema_200_value = data["Ema 200"].iloc[i]

        # Obtener el último precio de cierre
        ultimo_precio_cierre = data['Close'].iloc[i]

        # Si el precio está por encima de la EMA de 200, establecer la señal de compra en True
        if ultimo_precio_cierre < ema_200_value:
            data.at[i, "Ema 200 TF Short"] = True

    return data

# %%
ema_200_window = 200

# %%
ema_200_long(data, ema_200_window)

# %%
ema_200_short(data, ema_200_window)

# %%
def simular_operaciones_long_short(data, combinacion_bits):
    profit_total = 0
    estado = None  # 'long', 'short', None
    precio_entrada = 0

    for i in range(len(data)):
        # Determinar la acción basada en la combinación de bits
        # Señales para largos
        accion_long = (combinacion_bits & 1 and data['RSI_TF_Long'].iloc[i]) or \
                      (combinacion_bits & 2 and data['MACD_TF'].iloc[i]) or \
                      (combinacion_bits & 4 and data['EMA_cross_TF_Long'].iloc[i]) or \
                      (combinacion_bits & 8 and data['Ema 200 TF Long'].iloc[i])
        
        # Señales para cortos 
        accion_short = (combinacion_bits & 1 and data['RSI_TF_Short'].iloc[i]) or \
                       (combinacion_bits & 2 and data['MACD_TF_Short'].iloc[i]) or \
                       (combinacion_bits & 4 and data['EMA_cross_TF_Short'].iloc[i]) or \
                       (combinacion_bits & 8 and data['Ema 200 TF Short'].iloc[i])

        # Lógica combinada
        if accion_long and not accion_short and estado != 'long':
            if estado == 'short':  # Cerrar posición corta
                profit_total += precio_entrada - data['Close'].iloc[i]
            precio_entrada = data['Close'].iloc[i]
            estado = 'long'
        elif accion_short and not accion_long and estado != 'short':
            if estado == 'long':  # Cerrar posición larga
                profit_total += data['Close'].iloc[i] - precio_entrada
            precio_entrada = data['Close'].iloc[i]
            estado = 'short'

    # Cerrar cualquier posición abierta al final
    if estado == 'long':
        profit_total += data['Close'].iloc[-1] - precio_entrada
    elif estado == 'short':
        profit_total += precio_entrada - data['Close'].iloc[-1]

    return profit_total

# Evaluar las combinaciones
profits = [simular_operaciones_long_short(data, combinacion_bits) for combinacion_bits in range(16)]

# Identificar la combinación con el máximo profit
max_profit_index = np.argmax(profits)
mejor_combinacion_bits = "{:04b}".format(max_profit_index)

print(f"La mejor combinación de modelos es: {mejor_combinacion_bits}")


# %%
def evaluate_performance(data):
    # Calculate returns
    data['Returns'] = data['Close'].pct_change()
    daily_returns = data['Returns']
    return daily_returns

# %%
def optimization(trial):
    # Set hyperparameters to optimize each indicator
    rsi_window_long = trial.suggest_int('rsi_window_long', 1, 50)
    rsi_window_short = trial.suggest_int('rsi_window_short', 1, 50)

    macd_window_fast_long = trial.suggest_int('macd_window_fast_long', 1, 50)
    macd_window_slow_long = trial.suggest_int('macd_window_slow_long', 1, 50)
    macd_window_sign_long = trial.suggest_int('macd_window_sign_long', 1, 50)
    macd_window_fast_short = trial.suggest_int('macd_window_fast_short', 1, 50)
    macd_window_slow_short = trial.suggest_int('macd_window_slow_short', 1, 50)
    macd_window_sign_short = trial.suggest_int('macd_window_sign_short', 1, 50)

    ema_13_window_long = trial.suggest_int('ema_13_window_long', 1, 100)
    ema_48_window_long = trial.suggest_int('ema_48_window_long', 1, 100)
    ema_13_window_short = trial.suggest_int('ema_13_window_short', 1, 100)
    ema_48_window_short = trial.suggest_int('ema_48_window_short', 1, 100)

    ema_200_window_long = trial.suggest_int('ema_200_window_long', 100, 500)
    ema_200_window_short = trial.suggest_int('ema_200_window_short', 100, 500)

    opt_data = data.copy()
    opt_data['Returns'] = opt_data['Close'].pct_change()

    # Apply indicators (as per your existing code)
    if opt_data['Returns'].isnull().any() or not np.isfinite(opt_data['Returns']).all():
        # Return a large negative value to indicate failure
        return -np.inf

    opt_performance = evaluate_performance(opt_data)

    return opt_performance

# %%
# Run optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(optimization, n_trials = 10)

# Find best parameters
best_params = study.best_params

# Apply bet hyperparameters
opt_data = data.copy()
opt_data = rsi_long(opt_data, window = best_params['rsi_window_long'])
opt_data = rsi_short(opt_data, window = best_params['rsi_window_short'])

opt_data = macd_long(opt_data, window_fast = best_params['macd_window_fast_long'], window_slow = best_params['macd_window_slow_long'], window_sign = best_params['macd_window_sign_long'])
opt_data = macd_short(opt_data, window_fast = best_params['macd_window_fast_short'], window_slow = best_params['macd_window_slow_short'], window_sign = best_params['macd_window_sign_short'])

opt_data = ema_cross_long(opt_data, ema_13_window = best_params['ema_13_window_long'], ema_48_window = best_params['ema_48_window_long'])
opt_data = ema_cross_short(opt_data, ema_13_window = best_params['ema_13_window_short'], ema_48_window = best_params['ema_48_window_short'])

opt_data = ema_200_long(opt_data, ema_200_window = best_params['ema_200_window_long'])
opt_data = ema_200_short(opt_data, ema_200_window = best_params['ema_200_window_short'])

# Evaluate results
best_params = study.best_params

# Print the optimum value for each window
print(f"Optimal RSI Window (Long): {best_params['rsi_window_long']}")
print(f"Optimal RSI Window (Short): {best_params['rsi_window_short']}")

# Print the optimal window values for MACD
print(f"Optimal MACD Window (Fast - Long): {best_params['macd_window_fast_long']}")
print(f"Optimal MACD Window (Slow - Long): {best_params['macd_window_slow_long']}")
print(f"Optimal MACD Window (Signal - Long): {best_params['macd_window_sign_long']}")
print(f"Optimal MACD Window (Fast - Short): {best_params['macd_window_fast_short']}")
print(f"Optimal MACD Window (Slow - Short): {best_params['macd_window_slow_short']}")
print(f"Optimal MACD Window (Signal - Short): {best_params['macd_window_sign_short']}")

# Print the optimal window values for EMA crossover
print(f"Optimal EMA Cross Window (13 - Long): {best_params['ema_13_window_long']}")
print(f"Optimal EMA Cross Window (48 - Long): {best_params['ema_48_window_long']}")
print(f"Optimal EMA Cross Window (13 - Short): {best_params['ema_13_window_short']}")
print(f"Optimal EMA Cross Window (48 - Short): {best_params['ema_48_window_short']}")

# Print the optimal window values for EMA 200
print(f"Optimal EMA 200 Window (Long): {best_params['ema_200_window_long']}")
print(f"Optimal EMA 200 Window (Short): {best_params['ema_200_window_short']}")

# %%
# Store the optimal window values in variables
rsi_window_long_optimal = best_params['rsi_window_long']
rsi_window_short_optimal = best_params['rsi_window_short']

macd_window_fast_long_optimal = best_params['macd_window_fast_long']
macd_window_slow_long_optimal = best_params['macd_window_slow_long']
macd_window_sign_long_optimal = best_params['macd_window_sign_long']
macd_window_fast_short_optimal = best_params['macd_window_fast_short']
macd_window_slow_short_optimal = best_params['macd_window_slow_short']
macd_window_sign_short_optimal = best_params['macd_window_sign_short']

ema_13_window_long_optimal = best_params['ema_13_window_long']
ema_48_window_long_optimal = best_params['ema_48_window_long']
ema_13_window_short_optimal = best_params['ema_13_window_short']
ema_48_window_short_optimal = best_params['ema_48_window_short']

ema_200_window_long_optimal = best_params['ema_200_window_long']
ema_200_window_short_optimal = best_params['ema_200_window_short']

# %%
#rsi_long(opt_data, window=rsi_window_long_optimal)
#orsi_short(opt_data, window=rsi_window_short_optimal)

#macd_long(opt_data, window_fast=macd_window_fast_long_optimal, window_slow=macd_window_slow_long_optimal, window_sign=macd_window_sign_long_optimal)
#macd_short(opt_data, window_fast=macd_window_fast_short_optimal, window_slow=macd_window_slow_short_optimal, window_sign=macd_window_sign_short_optimal)

#ema_cross_long(opt_data, ema_13_window=ema_13_window_long_optimal, ema_48_window=ema_48_window_long_optimal)
#ema_cross_short(opt_data, ema_13_window=ema_13_window_short_optimal, ema_48_window=ema_48_window_short_optimal)

#ema_200_long(opt_data, ema_200_window=ema_200_window_long_optimal)
#ema_200_short(opt_data, ema_200_window=ema_200_window_short_optimal)

# %%
def backtest_optimized(data, capital, take_profit_pct=None, stop_loss_pct=None):
    # Apply the indicator functions to the data
    data = rsi_long(data, window)
    data = macd_long(data, window_slow, window_fast, window_sign)
    data = ema_cross_long(data, ema_13_window, ema_48_window)
    data = ema_200_long(data, ema_200_window)
    data = rsi_short(data, window)
    data = macd_short(data, window_slow, window_fast, window_sign)
    data = ema_cross_short(data, ema_13_window, ema_48_window)
    data = ema_200_short(data, ema_200_window)

    # Obtener la mejor combinación de bits
    profits = [simular_operaciones_long_short(data, combinacion_bits) for combinacion_bits in range(16)]
    max_profit_index = profits.index(max(profits))
    mejor_combinacion_bits = "{:04b}".format(max_profit_index)

    # Inicializar columnas para señales de compra, venta y compra a corto
    data["Buy_Signal"] = False
    data["Sell_Signal"] = False
    data["Short_Sell_Signal"] = False

    # Inicializar lista para registrar cambios en el capital con ceros
    capital_history = [capital] * len(data)

    # Iterar sobre los índices
    for i in range(len(data)):
        operar = False

        # Verificar si algún modelo activo da señal de compra o venta corta según la mejor combinación de bits
        if mejor_combinacion_bits[0] == '1' and data['RSI_TF_Long'].iloc[i]:
            operar = True
            data.at[i, "Buy_Signal"] = True
        if mejor_combinacion_bits[1] == '0' and data['MACD_TF'].iloc[i]:
            operar = True
            data.at[i, "Buy_Signal"] = True
        if mejor_combinacion_bits[2] == '0' and data['EMA_cross_TF_Long'].iloc[i]:
            operar = True
            data.at[i, "Buy_Signal"] = True
        if mejor_combinacion_bits[3] == '1' and data['Ema 200 TF Long'].iloc[i]:
            operar = True
            data.at[i, "Buy_Signal"] = True

        if mejor_combinacion_bits[0] == '1' and data['RSI_TF_Short'].iloc[i]:
            operar = True
            data.at[i, "Short_Sell_Signal"] = True
        if mejor_combinacion_bits[1] == '1' and data['MACD_TF_Short'].iloc[i]:
            operar = True
            data.at[i, "Short_Sell_Signal"] = True
        if mejor_combinacion_bits[2] == '1' and data['EMA_cross_TF_Short'].iloc[i]:
            operar = True
            data.at[i, "Short_Sell_Signal"] = True
        if mejor_combinacion_bits[3] == '1' and data['Ema 200 TF Short'].iloc[i]:
            operar = True
            data.at[i, "Short_Sell_Signal"] = True

        # Verificar si hay capital suficiente para realizar la operación
        if capital <= 0:
            operar = False

        # Verificar si hay que aplicar take profit o stop loss
        if operar:
            if take_profit_pct and data['Close'].iloc[i] >= data['Close'].iloc[i-1] * (1 + take_profit_pct):
                operar = False
                data.at[i, "Sell_Signal"] = True  # Generar señal de venta
            if stop_loss_pct and data['Close'].iloc[i] <= data['Close'].iloc[i-1] * (1 - stop_loss_pct):
                operar = False

        # Actualizar el capital según las operaciones
        if operar:
            # Calcular el monto a comprar
            amount_to_buy = min(capital * 0.1, capital)
            capital -= amount_to_buy
        else:
            # Cerrar la posición si el capital invertido supera el 10% del capital inicial
            if data['Buy_Signal'].iloc[i] or data['Short_Sell_Signal'].iloc[i]:
                capital += amount_to_buy
                amount_to_buy = 0

        # Actualizar el valor de capital_history
        capital_history[i] = capital

    # Agregar columna para el historial del capital al dataframe
    data["Capital"] = capital_history

    # Convertir columnas booleanas a enteros
    buy_signal_int = data["Buy_Signal"].astype(int)
    sell_signal_int = data["Sell_Signal"].astype(int)
    short_sell_signal_int = data["Short_Sell_Signal"].astype(int)

    # Calcular el valor del portafolio (capital + valor actual de todas las acciones en posesión)
    valor_portafolio = capital + ((data["Close"] * buy_signal_int) - (data["Close"] * sell_signal_int) - (data["Close"] * short_sell_signal_int)).sum()
    return capital, valor_portafolio, data


# %%
import locale

# Set the locale to the appropriate currency format
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Ejecutar la función backtest_optimized para obtener el historial del capital
take_profit_pct = 0.05
stop_loss_pct = 0.01
capital = 1000000
capital, valor_portafolio, data = backtest_optimized(data, capital)

# Obtener capital_history de los datos generados por backtest_optimized
capital_history = data["Capital"].tolist()

# Formatear el valor del portafolio
formatted_valor_portafolio = locale.currency(valor_portafolio, grouping=True)

# Imprimir el valor del portafolio
print(f"Profit using the optimized strategy: {valor_portafolio}")





