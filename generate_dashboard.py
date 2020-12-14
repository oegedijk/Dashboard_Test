from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.custom import *
from joblib import load
import pandas as pd
from sklearn.model_selection import train_test_split

# Load model & data
dec_tree = load('dec_tree_v3.joblib')
df_data = pd.read_csv('data_v3_enc.csv')

# Prepare & split data
X = df_data.drop(['Duration', 'Timestamp'], axis=1)
y = df_data.Duration
Xt, X_small, yt, y_small = train_test_split(X, y, test_size=0.01, random_state=0)

exp = RegressionExplainer(dec_tree, X_small, y_small, cats=['Day_of_week', 'Hour', 'Vehicle', 'Position'])

# Build
db = ExplainerDashboard(exp, [ShapDependenceComposite, WhatIfComposite], hide_whatifpdp=True)

# Save
exp.dump("explainer.joblib")
db.to_yaml("dashboard.yaml")
