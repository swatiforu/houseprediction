import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, redirect, url_for, flash, jsonify

app = Flask(__name__)

@app.route('/prediction/', methods=['GET'])
def getPredictions():
  beds = request.args['beds']
  baths = request.args['baths']
  size = request.args['size']
  location = request.args['location']
  return gethouseprice(beds, baths, size, location)

def gethouseprice(beds, baths, size, location):
  beds = int(beds)
  baths = int(baths)
  spl = size.split(' ')
  area = float(spl[0])
  if spl[1] == 'Kanal':
    area = area*20
  locs = pd.read_excel('locs.xlsx')
  val = locs[locs['Location'] == location].Index.values[0]
  df = pd.read_excel('Data.xlsx')
  df.drop(['Unnamed: 0'], axis = 1, inplace=True)
  unscaled_data = df.values.copy()
  X = []
  y = []
  for i in unscaled_data:
    X.append(i[:-1])
    y.append(np.asarray([i[-1]]))
  X = np.asarray(X)
  y = np.asarray(y)
  scaler = MinMaxScaler((0,1))
  X = scaler.fit_transform(X)
  scaler2 = MinMaxScaler((0,1))
  y = scaler2.fit_transform(y)
  regressor = pickle.load(open('House.pickle', 'rb'))
  k = np.asarray([area, baths, beds, val])
  values = scaler.transform(np.asarray([k]))
  result = regressor.predict(values)
  result = scaler2.inverse_transform(np.asarray([result]))
  return str(result[0][0])
