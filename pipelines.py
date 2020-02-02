from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from preprocessing import load_noshows_preprocessed

MODELS = {
    "tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "boosting": GradientBoostingClassifier,
    "svm": SVC,
    "neural": MLPClassifier,
    'lr': LogisticRegression
}

def make_pipeline(method="tree"):
    scaler = StandardScaler()
    model = MODELS[method]()
    pipe = Pipeline([("normalize", scaler), ("classify", model)])
    return pipe
