import os
from feature_generation import prepare_features, combine_stocks, ml_preprocessing
from models import RNNModel, CNNModel
from ploting import plot_data
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import pandas as pd

model_LinearRegression = LinearRegression()
model_RandomForest = RandomForestRegressor()
model_XGB = XGBRegressor(n_estimators=10, learning_rate=0.05, n_jobs=-1, random_state=0)

#create df for model comparison
df = pd.DataFrame(columns=[ 'Model', 'MAE_train', 'MAE_valid', 'MAE_test'])
# Aktien auswählen
relevant_stocks = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'BRK-B', 'LLY', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'MRK', 'ORCL', 'CVX', 'BAC', 'KO', 'CRM', 'NFLX', 'PEP', 'AMD', 'TMO', 'ADBE', 'WFC', 'LIN', 'QCOM', 'CSCO', 'MCD', 'ACN', 'DIS', 'DHR', 'ABT', 'INTU', 'GE', 'CAT', 'AMAT', 'AXP', 'TXN', 'VZ', 'AMGN', 'PFE', 'MS', 'CMCSA', 'IBM', 'NEE', 'UNP']

# Feature-Generation für jede Aktie
for symbol in relevant_stocks:
    file_path = os.path.join('data', 'stock_dataframes', f'{symbol}.csv')
    if not os.path.exists(file_path): # nicht nochmals erstellen wenn schon vorhanden
        prepare_features(symbol, option_volume=False)

# Zusammenführen, Clean-up NaN-Values (im Normalfall 1 Zeile), One-Hot Encoding der kategorischen Variable 'Sector', abspeichern 
stock_df = combine_stocks(relevant_stocks, return_df=True)

# Vorbereitung des Trainings-, Validierungs- und Testdatensatzes
X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test, df_train, df_valid, df_test = ml_preprocessing(stock_df)

print("Training Linear Regression...")
model_LinearRegression.fit(X_train_scaled, y_train)

y_pred_train = model_LinearRegression.predict(X_train_scaled)
y_pred_valid = model_LinearRegression.predict(X_valid_scaled)
y_pred_test = model_LinearRegression.predict(X_test_scaled)

print("Max Value in any column of X_train_scaled: ", X_train_scaled.max())
print("Min Value in any column of X_train_scaled: ", X_train_scaled.min())

###Results###
print("Linear Regression ----------------------------------------------------")
print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))
#save results in df
df = pd.concat([df, pd.DataFrame({'Model': ['Linear Regression'], 'MAE_train': [mean_absolute_error(y_train, y_pred_train)], 'MAE_valid': [mean_absolute_error(y_valid, y_pred_valid)], 'MAE_test': [mean_absolute_error(y_test, y_pred_test)]})], ignore_index=True)

plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='Linear Regression')
plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='Linear Regression')
plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='Linear Regression')


# SHAP for Linear Regression
explainer_lr = shap.Explainer(model_LinearRegression, X_train_scaled)
shap_values_lr = explainer_lr(X_train_scaled)

# Summary Plot
print("SHAP summary plot for Linear Regression")
shap.summary_plot(shap_values_lr, X_train_scaled)
plt.savefig('plots/explainer/lr_summary_plot.png')
plt.close()

# Force Plot for a single prediction
print("SHAP force plot for a single prediction - Linear Regression")
shap.force_plot(shap_values_lr[0], matplotlib=True)
plt.savefig('plots/explainer/lr_force_plot.png')
plt.close()

# Waterfall Plot for a single prediction
print("SHAP waterfall plot for a single prediction - Linear Regression")
shap.waterfall_plot(shap_values_lr[0])
plt.savefig('plots/explainer/lr_waterfall_plot.png')
plt.close()

###XGBoost###
print("Training XGBoost...")
model_XGB.fit(X_train_scaled, y_train)

y_pred_train = model_XGB.predict(X_train_scaled)
y_pred_valid = model_XGB.predict(X_valid_scaled)
y_pred_test = model_XGB.predict(X_test_scaled)

###Results###
print("XGBoost ----------------------------------------------------")
print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))

plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='XGBoost')

#df
df = pd.concat([df, pd.DataFrame({'Model': ['XGBoost'], 'MAE_train': [mean_absolute_error(y_train, y_pred_train)], 'MAE_valid': [mean_absolute_error(y_valid, y_pred_valid)], 'MAE_test': [mean_absolute_error(y_test, y_pred_test)]})], ignore_index=True)




# Reshape the data for RNN
X_train_np = X_train_scaled.to_numpy().reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_valid_np = X_valid_scaled.to_numpy().reshape(X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1)
X_test_np = X_test_scaled.to_numpy().reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

RNNModel = RNNModel(X_train_scaled.shape[1])
#compile model
RNNModel.model.compile(optimizer='adam', loss='mean_absolute_error')

history = RNNModel.model.fit(X_train_np, y_train, epochs=100, batch_size=1280, validation_data=(X_valid_np, y_valid))

y_pred_train = RNNModel.model.predict(X_train_scaled)
y_pred_valid = RNNModel.model.predict(X_valid_scaled)
y_pred_test = RNNModel.model.predict(X_test_scaled)

plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='AAPL', model='RNN')
plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='GOOGL', model='RNN')
plot_data(df_train, df_valid, df_test, y_pred_train, y_pred_valid, y_pred_test, symbol='MSFT', model='RNN')



print("RNN ----------------------------------------------------")
print("Mean absolute error on training set: ", mean_absolute_error(y_train, y_pred_train))
print("Mean absolute error on validation set: ", mean_absolute_error(y_valid, y_pred_valid))
print("Mean absolute error on test set: ", mean_absolute_error(y_test, y_pred_test))

#write results in df
df = pd.concat([df, pd.DataFrame({'Model': ['RNN'], 'MAE_train': [mean_absolute_error(y_train, y_pred_train)], 'MAE_valid': [mean_absolute_error(y_valid, y_pred_valid)], 'MAE_test': [mean_absolute_error(y_test, y_pred_test)]})], ignore_index=True)

#save history
pd.DataFrame(history.history).to_csv('history.csv')

#plot history
history = pd.read_csv('history.csv')
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('plots/histories/history_loss_rnn.png')
plt.close()

plt.plot(history['mae'], label='train')
plt.plot(history['val_mae'], label='validation')
#title
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend()
plt.savefig('plots/histories/history_mae_rnn.png')

print(df)
#plot model comparison
plt.figure(figsize=(10, 5))
plt.bar(df['Model'], df['MAE_train'], label='MAE_train')
plt.bar(df['Model'], df['MAE_valid'], label='MAE_valid')
plt.bar(df['Model'], df['MAE_test'], label='MAE_test')
plt.title('Model Comparison')
plt.ylabel('MAE')
plt.legend()
plt.savefig('plots/results/model_comparison.png')