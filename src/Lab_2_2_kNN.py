# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns
import matplotlib

def minkowski_distance(a, b, p=2):

    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    suma = 0
    for i in range(len(a)):
        suma += (abs(a[i]-b[i]))**p
        
    distancia = suma**(1/p)
    return distancia


# k-Nearest Neighbors Model
# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be equal.")
               
        elif k<0 or p<0:
            raise ValueError("k and p must be positive integers.")
        else:
            self.x_train = X_train
            self.y_train = y_train
            self.k = k
            self.p = p

        

    def predict(self, X: np.ndarray) -> np.ndarray:     
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        etiquetas = []
        for x in X:
            distancias = self.compute_distances(x)
            vecinosind = self.get_k_nearest_neighbors(distancias)
            vecinos = self.y_train[vecinosind]
            etiqueta = self.most_common_label(vecinos)
            etiquetas.append(etiqueta)
    
        return etiquetas 
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        probabilidades = []
        dicc = {}
        for x in X:
            probx = []
            distancias = self.compute_distances(x)
            vecinosind = self.get_k_nearest_neighbors(distancias)
            etiquetas = self.y_train[vecinosind]
            for label in etiquetas:
                if label not in dicc:
                    dicc[label] = 1
                else:
                    dicc[label] = dicc[label]+1
            for label in dicc:
                probx.append(dicc[label]/self.k)
            probabilidades.append(probx)

        return np.array(probabilidades)


    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        array = []
        for puntok in self.x_train:
            a = minkowski_distance(puntok,point,self.p)
            array.append(a)
        return array 
    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.
        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function
        """
        indices = np.argsort(distances)
        return indices[0:self.k]
    
    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        dicc = {}
        for label in knn_labels:
            if label not in dicc:
                dicc[label] = 1
            else:
                dicc[label] = dicc[label]+1
        max = 0
        comun = ""
        for item in dicc:
            if dicc[item]>max:
                max = dicc[item]
                comun = item
        return comun
            

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tn = 0
    fp = 0 
    fn = 0
    tp = 0

    for i in range(len(y_true_mapped)):
        if y_true_mapped[i] == y_pred_mapped[i]:
            if y_true_mapped[i] == 0:
                tn+= 1
            else:
                tp+= 1
        else:
            if y_true_mapped[i] == 0:
                fp+= 1
            else:
                fn+= 1

    # Accuracy
    if (tp + tn + fp + fn) == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    if (tp + fp) == 0:
        precision = 0
    else: 
        precision = tp/(tp + fp)
    # Recall (Sensitivity)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp/(tp + fn)
    # Specificity
    if (tn + fp) == 0:
        specificity = 0
    else:
        specificity = tn/(tn + fp)

    # F1 Score
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 =  2*(precision * recall) / (precision + recall)

    return {    
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }


def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
                              Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
        n_bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Center values of each bin.
            - "true_proportions": Fraction of positives in each bin.
    """
    y_true_m = np.array([1 if etiqueta == positive_label else 0 for etiqueta in y_true])

    intervalos = np.linspace(0, 1, n_bins + 1)
    indices_bins = np.digitize(y_probs, intervalos) - 1

    centros_bins = []
    proporciones_reales = []

    for i in range(n_bins):
        mascara = (indices_bins == i)
        if np.sum(mascara) == 0:
            fraccion_positivos = 0
        else:
            fraccion_positivos = np.mean(y_true_m[mascara])

        centro = 0.5 * (intervalos[i] + intervalos[i + 1])
        centros_bins.append(centro)
        proporciones_reales.append(fraccion_positivos)

    centros_bins = np.array(centros_bins)
    proporciones_reales = np.array(proporciones_reales)

    plt.figure(figsize=(5, 5))
    plt.plot(centros_bins, proporciones_reales)
    plt.show()

    return {
        "bin_centers": centros_bins,
        "true_proportions": proporciones_reales
    }



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label (int or str): The label considered as the positive class.
        n_bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        dict: A dictionary with:
            - "array_passed_to_histogram_of_positive_class"
            - "array_passed_to_histogram_of_negative_class"
    """

    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    positivos = y_probs[y_true_mapped == 1]
    negativos = y_probs[y_true_mapped == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(positivos, bins=n_bins, alpha=0.5,)
    plt.hist(negativos, bins=n_bins, alpha=0.5, color='red')
    plt.legend()
    plt.show()

    return {
            "array_passed_to_histogram_of_positive_class": positivos,
            "array_passed_to_histogram_of_negative_class": negativos}



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label (int or str): The label considered as the positive class.

    Returns:
        dict: Contains:
            - "fpr": Array of False Positive Rates for each threshold
            - "tpr": Array of True Positive Rates for each threshold
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    thresholds = np.linspace(0, 1, 11)

    tpr_list = []
    fpr_list = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))
   
        if (tp + fn) == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)

        if (tn + fp) == 0:
            fpr = 0
        else: 
            fpr = fp / (fp + tn)
            
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_list, tpr_list, marker='o', label='ROC')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return {
        "fpr": np.array(fpr_list),
        "tpr": np.array(tpr_list)
    }
