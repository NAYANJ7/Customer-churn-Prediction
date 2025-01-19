# %% [markdown]
# ### Importing Libraries

# %%
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTEENN

# %% [markdown]
# #### Reading csv

# %%
df=pd.read_csv("tel_churn.csv")
df.head()

# %%
df=df.drop('Unnamed: 0',axis=1)

# %%
x=df.drop('Churn',axis=1)
x

# %%
y=df['Churn']
y

# %% [markdown]
# ##### Train Test Split

# %%
sm=SMOTEENN(random_state=42)
x_train1,y_train1 = sm.fit_resample(x,y)
x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(x_train1,y_train1,test_size=0.2,random_state=42)

# %% [markdown]
# #### Decision Tree 
# because it can:
# Better handles non-linear relationships between features
# No feature scaling required
# Can automatically capture feature interactions

# %%
from sklearn.tree import DecisionTreeClassifier
model_log=DecisionTreeClassifier(max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    criterion='gini',
    class_weight='balanced',
    random_state=100)

# %%
model_log.fit(x_train_1,y_train_1)

# %%
y_pred=model_log.predict(x_test_1)
y_pred

# %%
model_log.score(x_test_1,y_test_1)

# %%
print(classification_report(y_test_1, y_pred, labels=[0,1]))

# %%
import pickle

# %%
filename = 'model.sav'

# %%
pickle.dump(model_log, open(filename, 'wb'))

# %%
load_model = pickle.load(open(filename, 'rb'))

# %%
load_model.score(x_test_1,y_test_1)


