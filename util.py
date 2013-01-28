from dateutil.parser import parse
import pandas as pd
import os

data_path = "C:/Users/Chris/Dropbox/Documents/kaggle/bulldozers/BulldozerPrices/data/"    
submission_path = "C:/Users/Chris/Dropbox/Documents/kaggle/bulldozers/BulldozerPrices/data/submissions/"

def get_train_df():
    train = pd.read_csv(data_path + "Train.csv", converters={"saledate": parse})
    return train 

def get_test_df():
    test = pd.read_csv(os.path.join(data_path, "Valid.csv"),
        converters={"saledate": parse})
    return test 

def get_train_test_df():
    return get_train_df(), get_test_df()

def write_submission(submission_name, predictions):
    test = get_test_df()    
    test = test.join(pd.DataFrame({"SalePrice": predictions}))

    test[["SalesID", "SalePrice"]].to_csv(submission_path + submission_name, index=False)