# import pandas as pd
# import numpy as np 
# df = pd.read_csv("C:/Users/vidya/Downloads/daily-renewable-energy-generation.csv")

# # Data preprocessing
# del df['id']
# df = df.dropna()
# df['date']=pd.to_datetime(df['date'])
# df.head()
# start_date='2018-01-01'
# end_date='2024-12-31'
# df=df[(df['date']>=start_date) & (df['date']<=end_date)]
# df.head()
# df.shape

# # Label Encoding
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# df['state_name'] = le.fit_transform(df['state_name'])
# df['sector'] = le.fit_transform(df['sector'])
# df['owner'] = le.fit_transform(df['owner'])
# df['type'] = le.fit_transform(df['type'])
# df['station'] = le.fit_transform(df['station'])

# # Removing outliers
# Q1 = df['actual_generation'].quantile(0.25)
# Q3 = df['actual_generation'].quantile(0.75)
# IQR=Q3-Q1
# outlier=df[((df['actual_generation'] < Q1 - 1.5 * IQR) | (df['actual_generation'] > Q3 + 1.5 * IQR))] 
# df=df.drop(outlier.index)
# df.shape

# # Feature scaling
# l2=['state_name','state_code','station','installed_capacity']
# from sklearn.preprocessing import MinMaxScaler
# mm=MinMaxScaler()
# for i in l2:
#     df[i]=mm.fit_transform(df[[i]])
# df.head(5)

# # Train test split
# x=df[['type','state_name','station','installed_capacity']]
# y=df['actual_generation']
# from sklearn.model_selection import train_test_split        
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=1, shuffle=True) 
# xtest.shape

# # Decision tree
# from sklearn.tree import DecisionTreeRegressor
# dt=DecisionTreeRegressor()
# dt.fit(xtrain,ytrain)
# print(dt.score(xtest,ytest)) 






# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import pickle

# Load your dataset
file_path = "C:/Users/vidya/Downloads/daily-renewable-energy-generation.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Feature Engineering: Extract time-based features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek
data['quarter'] = data['date'].dt.quarter

# Cyclical transformation for month and day_of_week
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Preprocess categorical columns
categorical_columns = ['state_name', 'station', 'sector', 'owner', 'type']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store each label encoder for later use

# Drop missing values in the target column
data_cleaned = data.dropna(subset=['actual_generation'])

# Handle missing values in the features
numeric_columns = ['installed_capacity']
imputer = SimpleImputer(strategy='mean')
data_cleaned[numeric_columns] = imputer.fit_transform(data_cleaned[numeric_columns])

# Features and target
X = data_cleaned[['state_name', 'station', 'sector', 'owner', 'type', 'installed_capacity', 'month_sin', 'month_cos', 
                  'day_of_week_sin', 'day_of_week_cos', 'year', 'quarter']]
y = data_cleaned['actual_generation']

# Scale numeric features
scaler = StandardScaler()

# Fit and transform on the same feature set to avoid mismatches
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use in the Flask app
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

# Save the trained model in HDF5 format
model.save('energy_generation_model.h5')  
print('Model saved as HDF5')

# Save individual label encoders for Flask app
with open('state_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoders['state_name'], f)

with open('station_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoders['station'], f)

with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoders['type'], f)
