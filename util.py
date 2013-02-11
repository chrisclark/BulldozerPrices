from dateutil.parser import parse
import pandas as pd
import os
import re
import numpy as np

data_path = "C:/Users/Chris/Dropbox/Documents/kaggle/bulldozers/BulldozerPrices/data/"    
submission_path = "C:/Users/Chris/Dropbox/Documents/kaggle/bulldozers/BulldozerPrices/data/submissions/"

FILE_TRAIN = "Train.csv"
FILE_TEST = "Valid.csv"
FILE_JOINED_TRAIN = "joined_train.csv"
FILE_JOINED_TEST = "joined_test.csv"
FILE_MACHINE_INFO = "Machine_appendix.csv"

train = None
test = None

def get_df(filename, converters=None):
    return pd.read_csv(os.path.join(data_path, filename), converters=converters)

def write_submission(submission_name, predictions):
    print "writing submission '" + submission_name + "'..."
    test_sub = get_df(FILE_TEST)
    test_sub = test_sub.join(pd.DataFrame({"SalePrice": predictions}))
    test_sub[["SalesID", "SalePrice"]].to_csv(submission_path + submission_name, index=False)
    print "done."
    
def saledate(date_column):
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
    return pd.DataFrame({"Stick_Length_IN": newCol}, index=target_col.index)

def Tire_Size(target_col):
    newCol = []
    for x in target_col:
        if type(x) == str:
            matchObj = re.match(r'[0-9]+(\.[0-9][0-9]?)?', x)
            if matchObj: newCol.append(float(matchObj.group()))
            else: newCol.append(np.NaN)
        else:
            newCol.append(np.NaN)
    return pd.DataFrame({"Tire_Size_IN": newCol}, index=target_col.index)

def Undercarriage_Pad_Width(target_col):
    newCol = []
    for x in target_col:
        if type(x) == str:
            matchObj = re.match(r'[0-9]+(\.[0-9][0-9]?)?', x)
            if matchObj: newCol.append(float(matchObj.group()))
            else: newCol.append(np.NaN)
        else:
            newCol.append(np.NaN)
    return pd.DataFrame({"Undercarriage_Pad_Width_IN": newCol}, index=target_col.index)

def YearMade(target_col):
    return pd.DataFrame({
        "YearMade_filled": target_col.map(lambda x: 1993.75 if x == 1000 else x),
        "YearMade_is_1000": target_col.map(lambda x: 1 if x == 1000 else 0)
    }, index=target_col.index)
    
def MachineHoursCurrentMeter(target_col):
    return pd.DataFrame({
        "MachineHoursCurrentMeter_is_nan": target_col.map(lambda x: 1 if x == 0 or np.isnan(x) else 0),
        "MachineHoursCurrentMeter_Filled": target_col.map(lambda x: 3946 if x == 0 or np.isnan(x) else x)
    }, index=target_col.index)

def BuildCategorical(target_col):
    s = np.unique(target_col.values)
    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    return pd.DataFrame({target_col.name + "_fea": target_col.map(mapping)})

def DropFromBoth(cols):
    global train
    global test
    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)

def AddFeature(featureFn, colName):
    global train
    global test
    train = train.join(featureFn(train[colName])).drop(colName, axis=1)
    test = test.join(featureFn(test[colName])).drop(colName, axis=1)

def featuresToDisk(description):
    global train
    global test
    print "writing " + description + "..."
    train.to_csv("train_" + description + ".csv", index=False)
    test.to_csv("test_" + description + ".csv", index=False)
    print "done."

def getTrainTestFeatureMatrix(build=True):
    if build:
        
        global train
        global test
        
        #read in the raw data
        train = get_df(FILE_TRAIN, {"saledate": parse})
        test = get_df(FILE_TEST, {"saledate": parse})
        machine_info = get_df(FILE_MACHINE_INFO)
        
        #preemptively drop overlapping columns and merge
        DropFromBoth(['YearMade','fiModelDesc','fiBaseModel','fiSecondaryDesc','fiModelSeries','fiModelDescriptor','fiProductClassDesc','ProductGroup','ProductGroupDesc'])
        train = train.merge(machine_info, on="MachineID")
        test = test.merge(machine_info, on="MachineID")
        DropFromBoth('ModelID_y') #drop overlapping join column
        
        print "joining done"
        
        #build special features
        AddFeature(YearMade, "MfgYear")
        AddFeature(saledate, "saledate")
        AddFeature(Stick_Length, "Stick_Length")
        AddFeature(Tire_Size, "Tire_Size")
        AddFeature(Undercarriage_Pad_Width, "Undercarriage_Pad_Width")
        AddFeature(MachineHoursCurrentMeter, "MachineHoursCurrentMeter")
        
        print "feature pass 1 done"
        
        #Build categoricals for any non-numeric columns
        for col in set(train.columns):
            if train[col].dtype == np.dtype('object'): AddFeature(BuildCategorical, col)
        
        featuresToDisk("finalfeatures")
    else:
        print "Reading feature matrix from disk..."
        train = pd.read_csv("fm_train.csv")
        test = pd.read_csv("fm_test.csv")
        print "done."
    return train, test