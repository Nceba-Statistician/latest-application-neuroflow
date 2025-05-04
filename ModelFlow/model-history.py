import streamlit, tensorflow, os, pandas, numpy, seaborn, pickle
from keras.api.models import Sequential, save_model, load_model 
from keras.api.layers import  Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from keras.api.callbacks import TensorBoard, LambdaCallback
from scipy import stats
from statsmodels import api
import datetime
from sklearn.impute import KNNImputer
from io import StringIO
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

   