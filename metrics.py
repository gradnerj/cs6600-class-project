import pandas as pd
import numpy as np

def get_metrics(model, X, y, name):
    metrics = {}
    metrics['Name'] = name
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