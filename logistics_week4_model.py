# Week 4 Logistics Predictive Modeling
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
df=pd.read_csv("logistics_week4_dataset.csv")
X=df[["Distance_km","Shipment_Volume","Transport_Mode","Region","Transport_Cost_INR"]]
y=df["Delivery_Time_days"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),["Transport_Mode","Region"])],remainder="passthrough")
model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=42))])
model.fit(X_train,y_train);pred=model.predict(X_test)
print("MAE",mean_absolute_error(y_test,pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,pred)))
print("R2",r2_score(y_test,pred))
print("CV MAE",-cross_val_score(model,X_train,y_train,cv=5,scoring="neg_mean_absolute_error").mean())
