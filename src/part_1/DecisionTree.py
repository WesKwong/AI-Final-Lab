import sys
from typing import Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")


# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


# node
class TreeNode:

    def __init__(self,
                 feature=None,
                 label=None,
                 threshold=None,
                 childs=None) -> None:
        self.feature = feature
        self.label = label
        self.threshold = threshold
        self.childs = childs


# model
class DecisionTreeClassifier:

    def __init__(self) -> None:
        self.tree = None
        self.continue_map = np.array([0, 2, 3, 6, 7, 10, 12, 13])
        self.n_types_map = np.array(
            [-1, 2, -1, -1, 4, 2, -1, -1, 2, 2, -1, 2, -1, -1, 4, 5])

    def __str__(self) -> str:
        tree_str = ""
        queue = [(self.tree, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node.label is not None:
                tree_str += f"{'  ' * depth}label: {node.label}\n"
            else:
                if node.threshold is None:
                    tree_str += f"{'  ' * depth}feature: {node.feature}\n"
                    for i, child in enumerate(node.childs):
                        queue.append((child, depth + 1))
                else:
                    tree_str += f"{'  ' * depth}feature: {node.feature}, threshold: {node.threshold}\n"
                    for i, child in enumerate(node.childs):
                        queue.append((child, depth + 1))
        return tree_str

    def __repr__(self) -> str:
        return self.__str__()

    def _if_all_same(self, y: np.ndarray) -> bool:
        judge = len(np.unique(y)) == 1
        return judge

    def _if_feat_none_or_A_all_same(self, X: np.ndarray, y: np.ndarray,
                                    A: np.ndarray) -> bool:
        if len(A) == 0:
            return True
        for feature in A:
            if not self._if_all_same(X[:, feature]):
                return False
        return True

    def _entropy(self, y: np.ndarray) -> np.float64:
        n = len(y)
        cnt = np.bincount(y)
        p = cnt / n
        # avoid log(0)
        p[p == 0] = 1
        return -np.sum(p * np.log2(p))

    def _get_thresholds(self, X: np.ndarray, feature: np.int64) -> np.ndarray:
        n_values = np.sort(np.unique(X[:, feature]))
        if len(n_values) == 1:
            return n_values
        thresholds = (n_values[1:] + n_values[:-1]) / 2
        return thresholds

    def _gain(self, X: np.ndarray, y: np.ndarray, feature: np.int64,
              ent_y: np.float64) -> Tuple[np.float64, Union[np.float64, None]]:
        n = len(y)
        ent_x = 0
        gain = np.float64(-1)
        threshold = None
        if feature in self.continue_map:
            thresholds = self._get_thresholds(X, feature)
            max_local_gain = -1
            for local_thre in thresholds:
                idx = X[:, feature] <= local_thre
                ent_x = len(y[idx]) / n * self._entropy(y[idx])
                ent_x += len(y[~idx]) / n * self._entropy(y[~idx])
                local_gain = ent_y - ent_x
                if local_gain > max_local_gain:
                    max_local_gain = local_gain
                    threshold = local_thre
            gain = max_local_gain
        else:
            for i in range(self.n_types_map[feature]):
                idx = X[:, feature] == i
                ent_x += len(y[idx]) / n * self._entropy(y[idx])
            gain = ent_y - ent_x
        return gain, threshold

    def _get_best_split(
            self, X: np.ndarray, y: np.ndarray,
            A: np.ndarray) -> Tuple[np.int64, Union[np.float64, None]]:
        ent_y = self._entropy(y)
        threshold = None
        max_idx = np.int64(-1)
        max_gain = -1
        for feature in A:
            gain, local_thre = self._gain(X, y, feature, ent_y)
            if gain > max_gain:
                if local_thre is None:
                    threshold = None
                else:
                    threshold = local_thre
                max_idx = feature
                max_gain = gain
        return max_idx, threshold

    def _get_max_cnt_label(self, y: np.ndarray) -> np.int64:
        return np.argmax(np.bincount(y))

    def _gen_node(self, X: np.ndarray, y: np.ndarray,
                  A: np.ndarray) -> TreeNode:
        if self._if_all_same(y):
            return TreeNode(label=y[0])
        if self._if_feat_none_or_A_all_same(X, y, A):
            return TreeNode(label=self._get_max_cnt_label(y))
        best_split_feature, threshold = self._get_best_split(X, y, A)

        if best_split_feature in self.continue_map and threshold is None:
            raise ValueError(
                f"threshold can not be None for continue feature: {best_split_feature}"
            )
        if best_split_feature not in self.continue_map and threshold is not None:
            raise ValueError(
                f"threshold should be None for discrete feature: {best_split_feature}"
            )
        if best_split_feature == -1:
            raise ValueError("best_split_feature should not be -1")

        if threshold is None:
            node = TreeNode(feature=best_split_feature,
                            childs=[None] *
                            self.n_types_map[best_split_feature])
            for i in range(self.n_types_map[best_split_feature]):
                child_idx = X[:, best_split_feature] == i
                X_child, y_child = X[child_idx], y[child_idx]
                if len(X_child) == 0:
                    node.childs[i] = TreeNode(label=self._get_max_cnt_label(y))
                    continue
                A_child = A[A != best_split_feature]
                child = self._gen_node(X_child, y_child, A_child)
                node.childs[i] = child
        else:
            node = TreeNode(feature=best_split_feature,
                            threshold=threshold,
                            childs=[None, None])
            for i in range(2):
                child_idx = X[:, best_split_feature] <= threshold
                X_child, y_child = X[child_idx], y[child_idx]
                if len(X_child) == 0:
                    node.childs[i] = TreeNode(label=self._get_max_cnt_label(y))
                    continue
                A_child = A[A != best_split_feature]
                child = self._gen_node(X_child, y_child, A_child)
                node.childs[i] = child
        return node

    def fit(self, X, y):
        # X: [n_samples_train, n_features],
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        A = np.array(range(X.shape[1]))
        self.tree = self._gen_node(X, y, A)

    def predict(self, X):
        # X: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        y = np.zeros(X.shape[0])
        # TODO:
        # 1. predict y based on the tree
        for i in range(X.shape[0]):
            node = self.tree
            while node.label is None:
                if node.threshold is None:
                    node = node.childs[int(X[i][node.feature])]
                else:
                    if X[i][node.feature] <= node.threshold:
                        node = node.childs[0]
                    else:
                        node = node.childs[1]
            y[i] = node.label
        # 2. return y
        return y


def load_data(
        datapath: str = './data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',]
    discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def main():
    logger.info("Decision Tree Classifier")
    logger.info("Loading data")
    X_train, X_test, y_train, y_test = load_data(
        './data/ObesityDataSet_raw_and_data_sinthetic.csv')
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(
        X_test), np.array(y_train), np.array(y_test)
    logger.info("Training model")
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    logger.info("Predicting")
    y_pred = clf.predict(X_test)
    logger.info(accuracy(y_test, y_pred))


if __name__ == "__main__":
    main()
