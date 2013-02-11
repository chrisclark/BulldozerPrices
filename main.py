from sklearn.ensemble import RandomForestRegressor
import util

train_fea, test_fea = util.getTrainTestFeatureMatrix(build=True)

#drop the "unknown"/index cols
#train_fea = train_fea.drop("Unnamed: 0", axis=1).drop("Unnamed: 0.1", axis=1)
#test_fea = test_fea.drop("Unnamed: 0", axis=1).drop("Unnamed: 0.1", axis=1)

print "building random forest..."
rf = RandomForestRegressor(n_estimators=50, n_jobs=1, min_samples_leaf=5)
rf.fit(train_fea.drop("SalePrice", axis=1), train_fea["SalePrice"])
print "done."

print "predicting..."
predictions = rf.predict(test_fea)
print "done."   

util.write_submission("random_forest_benchmark.csv", predictions)