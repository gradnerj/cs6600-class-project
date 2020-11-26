import pandas as pd
import numpy as np
import tensorflow as tf

def get_metrics(model, X, y, name):
    metrics = {}
    metrics['Name'] = name
    if type(model) == tf.python.keras.engine.training.Model:
        y_pred = model.predict(X)
        y_pred = np.where(y_pred>=0.7, 1, y_pred)
        y_pred = np.where(y_pred<0.7, 0, y_pred)
        cm = pd.crosstab(y_pred.ravel(), y.values.ravel(), rownames=['pred'], colnames=['actual'])
    else:    
        cm = pd.crosstab(model.predict(X).ravel(), y.values.ravel(),rownames=['pred'], colnames=['actual'])
    metrics['CM'] = cm
    metrics['Specificity'] = cm[0][0] / (cm[0][0]+cm[0][1])
    metrics['Sensitivity'] = cm[1][1] / (cm[1][1] + cm[1][0])
    metrics['Precision'] = cm[1][1] / (cm[1][1] + cm[0][1])
    metrics['Accuracy'] = (cm[1][1] + cm[0][0]) / (cm[1][1]+cm[0][0]+cm[1][0]+cm[0][1])
    return metrics

def print_metrics(metrics):
    for k,v in metrics.items():
        if k == 'Name':
            print(k + ':', v)
        elif k == 'CM':  
            print("Cost Matrix:",'\n')
            display(v)
        else:
            print(k + ':','\n')
            print(v, '\n')