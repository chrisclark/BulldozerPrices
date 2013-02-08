from sklearn.ensemble import RandomForestRegressor
import util

train_fea, test_fea = util.getTrainTestFeatureMatrix(build=False)

rf = RandomForestRegressor(n_estimators=50, n_jobs=1, min_samples_leaf=5)
rf.fit(train_fea.drop("SalePrice", axis=1), train_fea["SalePrice"])
predictions = rf.predict(test_fea)
util.write_submission("random_forest_benchmark.csv", predictions)
print "done."