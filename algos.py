from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import math
import statistics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import socket
from pandas_datareader import data as pdr
import yfinance as yf
import datetime

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

app=Flask(__name__)

vader = SentimentIntensityAnalyzer()

st_symbol= None

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/result", methods=['POST'])
def result():
    search_query = request.form.get('search', '')
    global st_symbol
    st_symbol = search_query
    
    return render_template('result.html')

@app.route("/know")
def know():
    return redirect(url_for('know.html'))

def calculate_accuracies(accuracy_arima, accuracy_lstm, accuracy_rf, accuracy_lr, pred_arima, pred_lstm, pred_rf, pred_lr, decision):
    return accuracy_arima, accuracy_lstm, accuracy_rf, accuracy_lr, pred_arima, pred_lstm, pred_rf, pred_lr, decision

@app.route("/home", methods=['GET', 'POST'])
def home():    
    def get_historical_from_csv(df):
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not set(required_columns).issubset(df.columns):
            print("CSV file is missing required columns.")
            return None

        df['Adj Close'] = df['Close'].shift(-7)

        df = df.dropna(subset=['Adj Close'])

        #print("Data Retrieval Successful (from CSV).")
        return df

    def ARIMA_ALGO(df):
        
        if 'Date' in df.columns:
            df = df.set_index('Date')
        def arima_model(train, test):

            history = [x for x in train]
            predictions = list()
            forecast_set = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)

            for t in range(7):
                model = ARIMA(history, order=(6, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                forecast_set.append(yhat)
                history.append(output)
            return predictions, forecast_set

        Quantity_date = df[['Close']]
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]

       

        predictions, forecast_set = arima_model(train, test)

        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(test,label='Actual Price')
        plt.plot(predictions,label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/arplot.png')
        
        arima_pred=round(forecast_set[0],2)
        error_arima = round(math.sqrt(mean_squared_error(test, predictions)),2)
        accuracy_arima = round((r2_score(test, predictions)*100),2)
        mean = statistics.mean(forecast_set)

        print(forecast_set)
        return arima_pred, error_arima, accuracy_arima, forecast_set, mean

    def LSTM_ALGO(df):
        dataset_train = df.iloc[0:int(0.8 * len(df)), :]
        dataset_test = df.iloc[int(0.8 * len(df)):, :]
        training_set = df.iloc[:, 3:4].values  

        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)  

        X_train = []
        y_train = []

        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 7:i, 0])
            y_train.append(training_set_scaled[i, 0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # .shape 0=row,1=col
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

        regressor = Sequential()

        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.1))

        # Add 2nd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 3rd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))

        # Add o/p layer
        regressor.add(Dense(units=1))

        # Compile
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Training
        regressor.fit(X_train, y_train, epochs=10, batch_size=32)

        # For lstm, batch_size=power of 2
        # Testing
        real_stock_price = dataset_test.iloc[:, 3:4].values
        dataset_total = df.iloc[:, 3:4]
        testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
        testing_set = testing_set.reshape(-1, 1)
        testing_set = sc.transform(testing_set)

        X_test = []
        y_test = []
        for i in range(7, len(testing_set)):
            X_test.append(testing_set[i - 7:i, 0])
            y_test.append(testing_set[i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)

        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/lstmplot.png')

        error_lstm = round(math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price)), 2)

        forecasted_stock_price = regressor.predict(X_forecast)
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
        lstm_pred = round(forecasted_stock_price[0, 0], 2)
        accuracy_lstm = round(r2_score(real_stock_price, predicted_stock_price) * 100, 2)
        mean = forecasted_stock_price.mean()
        print("LSTM Model Retrieval Successful..")
        return mean, error_lstm, accuracy_lstm

    def LIN_REG_ALGO(df):
        forecast_out = int(7)
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new = df[['Close', 'Close after n days']]

        y = np.array(df_new.iloc[:-forecast_out, -1])
        y = np.reshape(y, (-1, 1))
        X = np.array(df_new.iloc[:-forecast_out, 0:-1])

        X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

        X_train = X[0:int(0.8 * len(df)), :]
        X_test = X[int(0.8 * len(df)):, :]
        y_train = y[0:int(0.8 * len(df)), :]
        y_test = y[int(0.8 * len(df)):, :]

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        y_test_pred = clf.predict(X_test)

        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_test_pred, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/lrplot.png')       

        error_lr = round(math.sqrt(mean_squared_error(y_test, y_test_pred)), 2)

        forecast_set = clf.predict(X_to_be_forecasted)
        mean = forecast_set.mean()
        lr_pred = round(forecast_set[0, 0], 2)
        accuracy_lr = round(r2_score(y_test, y_test_pred) * 100, 2)

        print("LR Model Retrieval Successful..")
        return mean, error_lr, accuracy_lr

    def RF_ALGO(df):
        forecast_out = int(7)
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new = df[['Close', 'Close after n days']]

        X = np.array(df_new.iloc[:-forecast_out, :-1])
        y = np.array(df_new.iloc[:-forecast_out, -1])

        X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, :-1])

        X_train = X[0:int(0.8 * len(df)), :]
        X_test = X[int(0.8 * len(df)):, :]
        y_train = y[0:int(0.8 * len(df))]
        y_test = y[int(0.8 * len(df)):]

        param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}

        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_rf_model = grid_search.best_estimator_

        y_test_pred = best_rf_model.predict(X_test)

        error_rf = np.sqrt(mean_squared_error(y_test, y_test_pred))
        accuracy_rf = best_rf_model.score(X_test, y_test) * 100  # R-squared score in percentage

        plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_test_pred, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/rfplot.png')
        
        forecast_set = best_rf_model.predict(X_to_be_forecasted)
        mean = forecast_set.mean()

        print("Random Forest Model Retrieval Successful..")
        return mean, error_rf, accuracy_rf

    def recommendation(pos, neg, neut, quote_data, mean):
        today_stock = quote_data.iloc[-1:]
        if today_stock.iloc[-1]['Close'] < mean:
            if neg>pos and neg>neut:
                idea="FALL"
                decision="SELL"
            else:
                idea="RISE"
                decision="BUY"
        else:
            idea= "FALL"
            decision= "SELL"

        return idea, decision

    def get_financial_news(api_key, stock_symbol):
        base_url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": f"{stock_symbol} stock news",
            "sortBy": "publishedAt",
            "pageSize": 20
        }
        news_list = []
        pos = 0
        neg = 0
        neut = 0
        global_polarity = 0.0
        
        response = requests.get(base_url, params=params, verify=False)
        response.raise_for_status()
        news_data = response.json()

        if news_data["status"] == "ok":
            articles = news_data["articles"]
            for article in articles:
                title=article['title']
                
                news_list.append(title)
                compound = vader.polarity_scores(title)["compound"]
                global_polarity = global_polarity + compound
                
                if (compound > 0):
                    pos = pos + 1
                elif (compound < 0):
                    neg = neg + 1
                else:
                    neut = neut + 1
        else:
            print("Error in API response")

        
        if global_polarity >= 0:
            news_pol = "OVERALL POSITIVE"
        else:
            news_pol = "OVERALL NEGATIVE"

        print("Sentiment Analysis Retrieval Successful..")

        max_sentiment = max(pos, neg, neut)
    
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neut]
    
        explode = (0.1 if max_sentiment == pos else 0, 
               0.1 if max_sentiment == neg else 0, 
               0.1 if max_sentiment == neut else 0)
    
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    
    # Show the pie chart
        plt.axis('equal')  
        plt.savefig('static/piechart.png')

        return global_polarity, news_list, pos, neg, neut, news_pol

    stock_symbol=st_symbol
    stock_symbol=stock_symbol.upper()
    stock_type = request.args.get('type', '')
    news_api_key = 'efe1f77897c44a75a2ea2dee7476648b' 
    print(stock_type)
    print(stock_symbol)

    dfname=stock_symbol.lower()

    if stock_type=='nse':
        dfname=dfname+'n.csv'
    elif stock_type=='bse':
        dfname=dfname+'b.csv'

    print(dfname)
    print(stock_type)

    csv_file_path = dfname
    df = pd.read_csv(csv_file_path)

    if stock_type=='bse':
        df = df[['Date', 'Open Price','High Price','Low Price','Close Price','No. of Trades']].rename(columns={'Open Price': 'Open', 'High Price': 'High','Low Price':'Low','Close Price':'Close','No. of Trades':'Volume'})

    quote_data=df.dropna()
    historical_data = get_historical_from_csv(df)

    #print(historical_data)
    
    if historical_data is not None and not historical_data.empty:
        today_stock = historical_data.iloc[-1:]
        today_stock = today_stock.round(2)
        historical_data = historical_data.dropna()
        
    print(historical_data)

    pred_arima, error_arima, accuracy_arima, forecast_set, mean = ARIMA_ALGO(historical_data)
    print(accuracy_arima)

    pred_lstm, error_lstm, accuracy_lstm = LSTM_ALGO(historical_data)
    print(accuracy_lstm)

    pred_lr, error_lr, accuracy_lr= LIN_REG_ALGO(historical_data)
    print(accuracy_lr)

    pred_rf, error_rf, accuracy_rf = RF_ALGO(historical_data)
    print(accuracy_rf)   

    global_polarity, news_list, pos, neg, neut, news_pol = get_financial_news(news_api_key, stock_symbol)
    idea, decision = recommendation(pos, neg, neut, quote_data, mean)
    total_items = len(news_list)

    arima_accuracy, lstm_accuracy, rf_accuracy, lr_accuracy, arima_pred, lstm_pred, rf_pred, lr_pred, fdecision = calculate_accuracies(accuracy_arima, accuracy_lstm, accuracy_rf, accuracy_lr, pred_arima, pred_lstm, pred_rf, pred_lr, decision)

    arima_accuracy=round(arima_accuracy,2)
    lstm_accuracy=round(lstm_accuracy,2)
    rf_accuracy=round(rf_accuracy,2)
    lr_accuracy=round(lr_accuracy,2)

    arima_pred=round(arima_pred,2)
    lstm_pred=round(lstm_pred,2)
    rf_pred=round(rf_pred,2)
    lr_pred=round(lr_pred,2)

    #print(decision)

    # Return a valid response tuple
    return render_template("finalres.html", aar=arima_accuracy, alstm=lstm_accuracy, arf=rf_accuracy, alr=lr_accuracy, arima_pred=arima_pred, lstm_pred=lstm_pred, rf_pred=rf_pred, lr_pred=lr_pred, decision=decision, fin_head=news_pol, idea=idea, quote=stock_symbol ,total_items=total_items, news_list = news_list)


if __name__=="__main__":
    port = find_free_port()
    print(f"Starting server on port {port}")
    app.run(port=port)
