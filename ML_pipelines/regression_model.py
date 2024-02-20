"""
 Source : https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py

in this example, we are seeing what is the effect of transforming target variable in linear regression.
So, target variable will be transformed as
1. All the entries in the target variable will be non-negative value.
2. applying an exponential function to obtain non-linear targets which cannot be fitted using a simple linear model.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
# TransformedTargetRegressor:Meta-estimator to regress on a transformed target.
from sklearn.linear_model import RidgeCV
# RidgeCV:Ridge regression with built-in cross-validation.

from sklearn.metrics import PredictionErrorDisplay
# PredictionErrorDisplay:Visualization of the prediction error of a regression model.

# Creating the random data set which will have 10000 samples with added noise.
X, y = make_classification(
    n_samples=10000,
    n_redundant=10,
    noise = 10,
    random_state=10
)

# transforming the target variable.
# transform all the element as non-zero

min_element_in_target = y.min()
y = np.expm1(y + abs(min_element_in_target) / 200)
y_transform = np.log1p(y)

# Below we plot the probability density functions of the target before and after applying the logarithmic functions.
f, (ax0, ax1) = plt.subplots(1, 2)
plt.title("Plot of target variable after/Before transform.")
ax0.hist(y, bins=100, density=True)
ax0.set_xlim([0, 2000])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Target distribution")

ax1.hist(y_transform, bins=100, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

f.suptitle("Synthetic data", y=1.05)
plt.tight_layout()

# Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def compute_score(y_true, y_pred):
    return {
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "MedAE": f"{median_absolute_error(y_true, y_pred):.3f}",
    }


# This is based on the orginal target wothout transforming the target variable.
f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
ridge_cv = RidgeCV().fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)

# This is based on the orginal target with transforming the target variable.
ridge_cv_with_transform_target = TransformedTargetRegressor(
    regressor=RidgeCV(),
    func=np.log1p,
    inverse_func=np.expm1
).fit(X_train, y_train)

y_pred_ridge_with_trans_target = ridge_cv_with_transform_target.predict(X_test)

#===================================================================================
# PredictionErrorDisplay:Visualization of the prediction error of a regression model.

# This is the prediction for without transformed target variable.
PredictionErrorDisplay.from_predictions(
    y_true=y_test,
    y_pred=y_pred_ridge,
    kind="actual_vs_predicted",
    ax=ax1,
    scatter_kwargs={"alpha": 0.5},
)

# This is the prediction for with transformed target variable.
PredictionErrorDisplay.from_predictions(
    y_test,
    y_pred_ridge_with_trans_target,
    kind="actual_vs_predicted",
    ax=ax1,
    scatter_kwargs={"alpha": 0.5},
)

# Add the score in the legend of each axis
for ax, y_pred in zip([ax0, ax1], [y_pred_ridge, y_pred_ridge_with_trans_target]):
    for name, score in compute_score(y_test, y_pred).items():
        ax.plot([], [], " ", label=f"{name}={score}")
    ax.legend(loc="upper left")

ax0.set_title("Ridge regression \n without target transformation")
ax1.set_title("Ridge regression \n with target transformation")
f.suptitle("Synthetic data", y=1.05)
plt.tight_layout()
plt.show()