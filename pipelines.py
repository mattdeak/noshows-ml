from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

MODELS = {
    "tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "boosting": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
    "svm": SVC,
    "neural": MLPClassifier,
    'lr': LogisticRegression
}

def make_pipeline(method="tree"):
    scaler = StandardScaler()
    model = MODELS[method]()
    pipe = Pipeline([("normalize", scaler), ("classify", model)])
    return pipe
