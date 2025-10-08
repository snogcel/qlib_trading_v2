# qlib_trader

> https://www.geeksforgeeks.org/machine-learning/lightgbm-light-gradient-boosting-machine/

**LightGBM Core Parameters**

LightGBM’s performance is heavily influenced by the core parameters that control the structure and optimization of the model. Below are some of the key parameters:

    objective: Specifies the loss function to optimize during training. LightGBM supports various objectives such as regression, binary classification and multiclass classification.

    task: It specifies the task we wish to perform which is either train or prediction. The default entry is train.

    num_leaves: Specifies the maximum number of leaves in each tree. Higher values allow the model to capture more complex patterns but may lead to overfitting.

    learning_rate: Determines the step size at each iteration during gradient descent. Lower values result in slower learning but may improve generalization.

    max_depth: Sets the maximum depth of each tree.

    min_data_in_leaf: Specifies the minimum number of data points required to form a leaf node. Higher values help prevent overfitting but may result in underfitting.

    num_iterations: It specifies the number of iterations to be performed. The default value is 100.

    feature_fraction: Controls the fraction of features to consider when building each tree. Randomly selecting a subset of features helps improve model diversity and reduce overfitting.

    bagging_fraction: Specifies the fraction of data to be used for bagging (sampling data points with replacement) during training.

    L1 and L2: Regularization parameters that control L1 and L2 regularization respectively. They penalize large coefficients to prevent overfitting.

    min_split_gain: Specifies the minimum gain required to split a node further. It helps control the tree's growth and prevents unnecessary splits.

    categorical_feature : It specifies the categorical feature used for training model.
