from random_bayes import RandomBayes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

with open("column_names.txt") as f:
    names = f.read()
names = names.split(',')

df = pd.read_csv('agaricus-lepiota.data', header=None, na_values='?')
df.columns = names
df = df.dropna()
df = df.drop('veil-type', axis=1)
names.remove('veil-type')

X_df = df[names[1:]]
y_df = df[names[:1]]

X = X_df.values
y = y_df.values

def process(c):
    print(c)
    rb_list = []
    
    num_iterations = 10

    for _ in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        model_rb = RandomBayes(n_submodels=c[0], n_features=c[1], smooth=c[2], bootstrap=c[3])
        model_rb.fit(X_train, y_train)
        rb_list.append(accuracy_score(y_test, model_rb.predict(X_test)))
        
        
    rb_mean = np.mean(np.array(rb_list))
    rb_std = np.std(np.array(rb_list))
    #random_bayes_results[c] = (rb_mean, rb_std)
    return [c, (rb_mean, rb_std)]