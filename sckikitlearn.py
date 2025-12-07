import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


df_ = datasets.load_iris()
df = pd.DataFrame()
df['sepal_length'] = df_['data'][:,0]
df['sepal_width'] = df_['data'][:,1]
df['target'] = df_['target'] == 1

#Define train and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[['sepal_length', 'sepal_width']],
                                        df['target'],
                                        test_size=0.2,
                                        random_state=42)


X_train.to_csv(r'input.txt', header=None, index=None, sep=' ', mode='a')
y_train.to_csv(r'out.txt', header=None, index=None, sep=' ', mode='a')
X_test.to_csv(r'tinput.txt', header=None, index=None, sep=' ', mode='a')
y_test.to_csv(r'toutput.txt', header=None, index=None, sep=' ', mode='a')


_, ax1 = plt.subplots()
scatter = ax1.scatter(df['sepal_length'], df['sepal_width'], c=df['target'])
ax1.set(xlabel= 'sepal_length', ylabel='sepal_width')
_ = ax1.legend(
    scatter.legend_elements()[0], ['Not versicolor', 'Versicolor'], loc="lower right", title="Species")

X_train = scaler.fit_transform(X_train)
y_train = y_train.to_numpy()
y_train = y_train.astype(int)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()


def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True):
    # Train the SVC
    # Instantiate the classifier model
    svm_classifier = svm.SVC(kernel=kernel, gamma=2).fit(X_train, y_train)
    # Predict new intances classes
    y_predicted = svm_classifier.predict(X_train)
    # Evaluate model's accuracy
    accuracy = sklearn.metrics.accuracy_score(y_train, y_predicted, normalize=False)
    print(f"{accuracy} / {np.size(X_train, 0)} = {accuracy/np.size(X_train, 0)*100} %")


    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -1.83962751, 2.30486738, -2.37377751, 2.99237573
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": svm_classifier, "X": X_train, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )


    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            svm_classifier.support_vectors_[:, 0],
            svm_classifier.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    plt.show()

plot_training_data_with_decision_boundary("linear")
plot_training_data_with_decision_boundary("rbf")