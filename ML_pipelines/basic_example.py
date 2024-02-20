# Source : https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

# get the dataset
from sklearn.datasets import make_classification

# Split the dataset for training and testing
from sklearn.model_selection import train_test_split

# preprocessing the datasets
from sklearn.preprocessing import StandardScaler

# get the SVM classification method
from sklearn.svm import SVC

# make the pipeline for training
from sklearn.pipeline import Pipeline

X, y = make_classification(random_state=0) # getting the data [toy dataset for classification]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0) # splitting the data

#==============================

# Pipeline takes list of steps in tuple (name, transform) ex: 'scaler' is the name and StandardScaler() is the transform
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())], verbose=True) # making pipeline- using standardization

#==============================
pipe.fit(X_train, y_train) # training the pipeline/execute the pipeline

Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])


# print("Pipe structure:\n", pipe)
# print("Pipe classes:\n", pipe.classes_)
# print("After training the score:", pipe.score(X_test, y_test))
# print("pipe named_steps [Scaler]:", pipe['scaler'])
# print("pipe named_steps [SVC]:", pipe['svc'])
# print("pipe n_feature_in:Number of features seen during first step fit method.")
# print(pipe.n_features_in_)

# print(pipe.decision_function(X)) # Transform the data, and apply `decision_function` with the final estimator.
                                # Call `transform` of each transformer in the pipeline. The transformed
                                # data are finally passed to the final estimator that calls
                                # `decision_function` method. Only valid if the final estimator
                                # implements `decision_function`.

# print(pipe.fit_predict(X_train, y_train)) # transforms the data and apply fit_predict with the final estimator

# print(pipe.get_metadata_routing()) # Get metadata routing of this object.
# print(pipe.get_params()) # get parameter of the pipeline.

