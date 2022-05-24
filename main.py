import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


nl_slope = pd.read_csv("/Users/V/Desktop/project_landslides/dem/nonlandslides_slope4.csv")
nl_aspect = pd.read_csv("/Users/V/Desktop/project_landslides/dem/nonlandslides_aspect4.csv")
l_slope = pd.read_csv("/Users/V/Desktop/project_landslides/dem/landslides_Talim_random_slope.csv")
l_aspect = pd.read_csv("/Users/V/Desktop/project_landslides/dem/landslides_Talim_random_aspect.csv")

nl_slope = nl_slope.drop(columns=['fid', 'id'])
nl_aspect = nl_aspect.drop(columns=['fid', 'id'])
l_slope = l_slope.drop(columns=['fid', 'id'])
l_aspect = l_aspect.drop(columns=['fid', 'id'])

nonlandslides = {'slope': nl_slope['SAMPLE_1'], 'aspect': nl_aspect['SAMPLE_1'], 'y': 0}
landslides = {'slope': l_slope['SAMPLE_1'], 'aspect': l_aspect['SAMPLE_1'], 'y': 1}
nonlandslides = pd.DataFrame(data=nonlandslides)
landslides = pd.DataFrame(data=landslides)

nonlandslides = nonlandslides[nonlandslides.slope > 9]
nonlandslides = nonlandslides[nonlandslides.slope < 45]
print(len(nonlandslides))

data = pd.concat([nonlandslides, landslides])
data = data.reset_index()
data = data.iloc[np.random.permutation(len(data))]
y = data['y']
data['n'] = [1 if data['aspect'].iloc[i] < 22.5 or data['aspect'].iloc[i] >= 337.5 else 0 for i in range(data.shape[0])]
data['ne'] = [1 if 22.5 <= data['aspect'].iloc[i] < 67.5 else 0 for i in range(data.shape[0])]
data['e'] = [1 if 67.5 <= data['aspect'].iloc[i] < 112.5 else 0 for i in range(data.shape[0])]
data['se'] = [1 if 112.5 <= data['aspect'].iloc[i] < 157.5 else 0 for i in range(data.shape[0])]
data['s'] = [1 if 157.5 <= data['aspect'].iloc[i] < 202.5 else 0 for i in range(data.shape[0])]
data['sw'] = [1 if 205.5 <= data['aspect'].iloc[i] < 247.5 else 0 for i in range(data.shape[0])]
data['w'] = [1 if 247.5 <= data['aspect'].iloc[i] < 292.5 else 0 for i in range(data.shape[0])]
data['nw'] = [1 if 292.5 <= data['aspect'].iloc[i] < 337.5 else 0 for i in range(data.shape[0])]
data.drop(columns=['index', 'y', 'aspect'], inplace=True)
print(y)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=1, penalty='l2', C=0.01, solver='lbfgs', max_iter=1000)
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, penalty='l2', C=0.01, solver='lbfgs'))
print(np.mean(cross_val_score(model, data, y, cv=10, scoring='roc_auc')))
print(cross_val_score(model, data, y, cv=10, scoring='roc_auc'))
print(np.mean(cross_val_score(model, data, y, cv=10, scoring='accuracy')))
print(cross_val_score(model, data, y, cv=10, scoring='accuracy'))

print(np.mean(cross_val_score(pipe, data, y, cv=10, scoring='roc_auc')))
print(cross_val_score(pipe, data, y, cv=10, scoring='roc_auc'))
print(np.mean(cross_val_score(pipe, data, y, cv=10, scoring='accuracy')))
print(cross_val_score(pipe, data, y, cv=10, scoring='accuracy'))

model.fit(data, y)
print(model.score(data, y))

print(model.coef_)
print(model.intercept_)

