import pandas as pd

data = pd.read_csv("Data/enjoysport.csv")

def ID3(data):
    if len(data['enjoysport'].unique()) == 1:
        return data['enjoysport'].iloc[0]
    if len(data.columns) == 1:
        return data['enjoysport'].mode().iloc[0]
    best_feature = data.columns[0]
    tree = {best_feature: {value: ID3(data[data[best_feature] == value].drop(columns=[best_feature])) for value in data[best_feature].unique()}}
    return tree

def classify(tree, sample):
    feature = list(tree.keys())[0]
    return tree[feature].get(sample[feature], data['enjoysport'].mode().iloc[0]) if isinstance(tree[feature], dict) else tree[feature]

decision_tree = ID3(data)
new_sample = {'sky': 'sunny', 'airtemp': 'warm', 'humidity': 'high', 'wind': 'strong', 'water': 'warm', 'forcast': 'same'}
classification_result = classify(decision_tree, new_sample)
print("Classification Result:", classification_result)
