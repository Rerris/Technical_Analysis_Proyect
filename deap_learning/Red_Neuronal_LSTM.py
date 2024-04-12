# %% [markdown]
# ## Neural Network

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 

#importar mse
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# %%
data=pd.read_csv("https://raw.githubusercontent.com/Rerris/Technical_Analysis_Proyect/9a7d477bbed852e752101fe85ac928200ae7ebd5/data/aapl_1d_train.csv")
data.dropna(inplace=True)

# %%
# To create values of  (X) y label (Y)
X = pd.DataFrame()
X["Pt"] = data.Close
for i in range(1, 31):  #This code creates the columns that represent the previous days desired
    X[f"Pt_{i}"] = data.Close.shift(i)
Y = data.Close.shift(-1)

# %%

X = X.iloc[30:-1]
Y = Y.iloc[30:-1]

# Split train and test
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

# %%
# Build the LSTM model
model = Sequential([
    LSTM(units=256, return_sequences=True, input_shape=(31, 1)),
    LSTM(units=128),
    Dense(units=1)
])



# %%
model.compile(optimizer=Adam(), loss= MeanSquaredError(), metrics=[RootMeanSquaredError()])


# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2, batch_size=64, callbacks=[early_stopping])

# %%
# Make predictions
train_predictions = model.predict(X_train.values.reshape(-1, 31, 1))
test_predictions = model.predict(X_test.values.reshape(-1, 31, 1))


# %%
# Plotting the results
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
predicted_prices = np.concatenate([train_predictions, test_predictions])

# Prediction
fig.add_trace(go.Scatter(x=data.index, y=predicted_prices.flatten(), mode='lines', name='Predicted Price',line=dict(color='blue')))

# Labels
fig.update_layout(title='Stock Price Prediction using LSTM',
                   xaxis_title='Time',
                   yaxis_title='Stock Price',
                   showlegend=True,
                   legend=dict(x=0, y=1))

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

# %%
data_test=pd.read_csv('https://raw.githubusercontent.com/Rerris/Technical_Analysis_Proyect/9a7d477bbed852e752101fe85ac928200ae7ebd5/data/aapl_1d_test.csv')
data_test.dropna(inplace=True)

# %%
X=pd.DataFrame()
X["Pt"] = data_test.Close
for i in range(1, 31):  # Generar características para desplazamientos de 1 a 30 días
    X[f"Pt_{i}"] = data_test.Close.shift(i)
Y = data_test.Close.shift(-1)
X_test=X

# %%
Y=data_test.Close.shift(-1)
Y.head(10)

# %%
test_predictions = model.predict(X_test.values.reshape(-1, 31, 1))

# %%
# Plot
fig = go.Figure(data=[go.Candlestick(x=data_test.index,
                open=data_test['Open'],
                high=data_test['High'],
                low=data_test['Low'],
                close=data_test['Close'])])

# Prediction
fig.add_trace(go.Scatter(x=data_test.index, y=test_predictions.flatten(), mode='lines', name='Predicted Price',line=dict(color='blue')))

# Labels
fig.update_layout(title='Stock Price Prediction using LSTM',
                   xaxis_title='Time',
                   yaxis_title='Stock Price',
                   showlegend=True,
                   legend=dict(x=0, y=1))

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# %%
from tensorflow.keras import models
# Save the Keras model
models.save_model(model, 'model.keras')



# %%
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model.keras')

# Initialize the portfolio
portfolio_value = 100000  # Initial portfolio value
cash = portfolio_value
stock = 0
history = []  # To keep track of buy/sell/hold actions

# Define other necessary variables for backtesting
time_steps = 1
take_profit = 1.01
stop_loss = 0.99
buying_price = 0  # Initialize the buying price
action = 'Hold'  # Initialize the action

# Perform predictions and execute backtesting
for i in range(len(X_test) - time_steps):
    # Prepare the test data and reshape it
    X_test_reshaped = X_test.iloc[i:i+time_steps].values.reshape(-1, 31, 1)
    
    # Predict the price of the next day
    predicted_price = model.predict(X_test_reshaped)[0][0]
    
    # Get the actual price of the next day
    actual_price = X_test.iloc[i+time_steps, 0]
    
    # Decide to buy, sell, or hold
    if predicted_price > actual_price:  # If the model predicts that the price will go up
        if cash > 0:
            # Buy only 5% of the portfolio value
            buy_value = min(cash, portfolio_value * 0.05)
            stock += buy_value / actual_price
            cash -= buy_value
            action = 'Buy'
            buying_price = actual_price
    elif predicted_price < actual_price or actual_price >= buying_price * take_profit or actual_price <= buying_price * stop_loss:  # If the model predicts that the price will go down or the take profit or stop loss is reached
        if stock > 0:
            # Sell only 5% of the portfolio value
            sell_value = min(stock * actual_price, portfolio_value * 0.05)
            stock -= sell_value / actual_price
            cash += sell_value
            action = 'Sell'
    else:
        action = 'Hold'
    
    # Calculate the portfolio value
    portfolio_value = cash + stock * actual_price
    
    # Save the action in the history
    history.append({
        'Day': i,
        'Action': action,
        'Cash': cash,
        'Stock': stock,
        'Portfolio Value': portfolio_value
    })

# Convert the history into a DataFrame
df_history = pd.DataFrame(history)


# %%
df_history

# %%
# Graficar el historial de la cartera
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_history['Day'], y=df_history['Portfolio Value'], mode='lines', name='Portfolio Value', line=dict(color='blue')))
fig.update_layout(title='Portfolio Value Over Time',
                   xaxis_title='Day',
                   yaxis_title='Portfolio Value',
                   showlegend=True,
                   legend=dict(x=0, y=1))
fig.show()

# %%
# Guardar la grafica como png
fig.write_image("portfolio_value_1D.png")


