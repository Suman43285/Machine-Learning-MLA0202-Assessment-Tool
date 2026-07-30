import pandas as pd
from math import log2
data = {
    'Outlook':['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
               'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                   'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':['High','High','High','High','Normal','Normal','Normal',
                'High','Normal','Normal','Normal','High','Normal','High'],
    'Wind':['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
            'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'Play':['No','No','Yes','Yes','Yes','No','Yes',
            'No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)
def entropy(target):
    values = target.value_counts(normalize=True)
    return -sum(p * log2(p) for p in values)
def information_gain(df, attribute, target='Play'):
    total_entropy = entropy(df[target])
    weighted_entropy = 0
    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy
print("Entropy of Dataset:", entropy(df['Play']))
print("\nInformation Gain")
best_attribute = None
best_gain = -1
for col in df.columns[:-1]:
    gain = information_gain(df, col)
    print(col, ":", round(gain,4))
    if gain > best_gain:
        best_gain = gain
        best_attribute = col
print("\nBest Attribute:", best_attribute)
print("Highest Information Gain:", round(best_gain,4))

