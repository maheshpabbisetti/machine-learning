import numpy as np
import pandas as pd

data = pd.read_csv('Data/enjoysport.csv')
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in specific_h] for _ in specific_h]

    for i, h in enumerate(concepts):
        for x in range(len(specific_h)):
            if target[i] == "yes" and h[x] != specific_h[x]:
                specific_h[x] = general_h[x][x] = '?'
        
            if target[i] == "no":
                general_h[x][x] = specific_h[x] if h[x] != specific_h[x] else '?'

        print("Steps of Candidate Elimination Algorithm", i + 1)
        print(specific_h)
        print(general_h)

    general_h = [h for h in general_h if h != ['?'] * len(specific_h)]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
