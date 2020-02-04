import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import re
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.__version__

# Importing the dataset
dataset = pd.read_csv('customers_data.csv')

df = pd.DataFrame(dataset)

snake_columns = []
for col in df.columns:
    snake_columns.append(re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower())
df.columns = snake_columns

X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

for col in ['num_of_products', 'has_cr_card', 'is_active_member']:
    X[col] = X[col].astype(bool)
    

colT = ColumnTransformer(
    [("dummy_col", OneHotEncoder(drop='first', categories=[['France', 'Spain', 'Germany'],
                                             ['Female', 'Male']]), [1,2]),
     ("norm", StandardScaler(), [0, 3, 4, 5, 9])],remainder='passthrough')

X = colT.fit_transform(X)
   
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = Sequential()

classifier.add(Dense(6, activation='relu', input_dim = 11))
classifier.add(Dense(6))
classifier.add(Dense(1, activation = 'tanh'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

y_pred = classifier.predict(X_test)
y_pred_yes = y_pred > 0.5

    
cm = confusion_matrix(y_test, y_pred_yes)

accuracy = accuracy_score(y_test, y_pred_yes)
# accuracy for this data is 0.842
