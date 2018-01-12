import pandas as pd
import sklearn as skl
from sklearn import svm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import cluster
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

h5data = pd.HDFStore('../data/raw_data.h5')

raw_data= h5data["raw_data"]

label = raw_data["label"]
data = raw_data.drop(["label"], axis=1)

np_data = data.as_matrix()
np_label = label.as_matrix()


x_train, x_test, y_train, y_test = train_test_split(np_data, np_label, test_size=0.2, random_state=42)



""" 
#SVM model
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')
svc_model.fit(x_train, y_train)
prediction = svc_model.predict(x_test)

"""


# Grid search for parameters
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)


#clf = cluster.KMeans(init='k-means++', n_clusters=5, random_state=42)
#clf.fit(x_train)
#prediction = clf.fit_predict(x_train)

fig, ax = plt.subplots(2, 2, figsize=(8, 4))

print("Start Down Scale")
from sklearn.decomposition import PCA
train_iso = PCA(n_components=2).fit_transform(x_train)
test_iso = PCA(n_components=2).fit_transform(x_test)

ax[0][0].scatter(train_iso[:, 0], train_iso[:, 1], c=y_train)
ax[0][0].set_title('Predicted Training Labels')

ax[1][0].scatter(test_iso[:, 0], test_iso[:, 1], c=y_test)
ax[1][0].set_title('Actual Test Labels')
ax[1][1].scatter(test_iso[:, 0], test_iso[:, 1], c=prediction)
ax[1][1].set_title('Prediction Test Labels')

plt.show()




# Evaluation
#accuracy = np.sum(np.equal(prediction, y_test))/len(prediction)
#print(confusion_matrix(y_test, prediction, labels=["0","1","2","3", "4"]))

