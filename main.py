import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import util
import re

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)

def Stick_Length(target_col):
    newCol = []
    for x in target_col:
        if type(x) == str and len(x.split(' ')) == 2:
            parts = x.replace('"',"").replace("'","").split(' ')
            parts = map(float, parts)
            newCol.append(parts[0] + (parts[1]/12.0))
        else:
            newCol.append(np.NaN)
    return pd.DataFrame({"Stick_Length": newCol}, index=target_col.index)

def Tire_Size(target_col):
    newCol = []
    for x in target_col:
        if type(x) == str:
            matchObj = re.match(r'[0-9]+(\.[0-9][0-9]?)?', x)
            if matchObj: newCol.append(float(matchObj.group()))
            else: newCol.append(np.NaN)
        else:
            newCol.append(np.NaN)
    return pd.DataFrame({"Tire_Size": newCol}, index=target_col.index)

def Undercarriage_Pad_Width(target_col):
    newCol = []
    for x in target_col:
        if type(x) == str:
            matchObj = re.match(r'[0-9]+(\.[0-9][0-9]?)?', x)
            if matchObj: newCol.append(float(matchObj.group()))
            else: newCol.append(np.NaN)
        else:
            newCol.append(np.NaN)
    return pd.DataFrame({"Undercarriage_Pad_Width": newCol}, index=target_col.index)

def YearMade(target_col):
    return pd.DataFrame({
        "YearMade": target_col.map(lambda x: 1993.75 if x == 1000 else x),
        "YearMade_is_1000": target_col.map(lambda x: 1 if x == 1000 else 0)
    }, index=target_col.index)
    
def MachineHoursCurrentMeter(target_col):
    return pd.DataFrame({
        "MachineHoursCurrentMeter_is_nan": target_col.map(lambda x: 1 if x == 0 or np.isnan(x) else 0),
        "MachineHoursCurrentMeter": target_col.map(lambda x: 3946 if x == 0 or np.isnan(x) else x)
    }, index=target_col.index)

train, test = util.get_train_test_df()

columns = set(train.columns)
columns.remove("SalesID")
columns.remove("SalePrice")
columns.remove("saledate")
columns.remove("Stick_Length")
columns.remove("Tire_Size")
columns.remove("Undercarriage_Pad_Width")
columns.remove("YearMade")
columns.remove("MachineHoursCurrentMeter")

train_fea = get_date_dataframe(train["saledate"])
train_fea = train_fea.join(Stick_Length(train["Stick_Length"]))
train_fea = train_fea.join(Tire_Size(train["Tire_Size"]))
train_fea = train_fea.join(Undercarriage_Pad_Width(train["Undercarriage_Pad_Width"]))
train_fea = train_fea.join(YearMade(train["YearMade"]))
train_fea = train_fea.join(MachineHoursCurrentMeter(train["MachineHoursCurrentMeter"]))

test_fea = get_date_dataframe(test["saledate"])
test_fea = test_fea.join(Stick_Length(test["Stick_Length"]))
test_fea = test_fea.join(Tire_Size(test["Tire_Size"]))
test_fea = test_fea.join(Undercarriage_Pad_Width(test["Undercarriage_Pad_Width"]))
test_fea = test_fea.join(YearMade(test["YearMade"]))
test_fea = test_fea.join(MachineHoursCurrentMeter(test["MachineHoursCurrentMeter"]))

for col in columns:
    if train[col].dtype == np.dtype('object'):
        s = np.unique(train[col].values)
        mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
        train_fea = train_fea.join(train[col].map(mapping))
        test_fea = test_fea.join(test[col].map(mapping))
    else:
        train_fea = train_fea.join(train[col])
        test_fea = test_fea.join(test[col])

train_fea.to_csv("fm_train.csv")
test_fea.to_csv("fm_test.csv")

rf = RandomForestRegressor(n_estimators=50, n_jobs=1, min_samples_leaf=5)
rf.fit(train_fea, train["SalePrice"])
predictions = rf.predict(test_fea)
util.write_submission("random_forest_benchmark.csv", predictions)
print "done."