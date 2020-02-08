
import pandas as pd

def load_intention():
    train = pd.read_csv("data/online_shoppers_intention.csv")
    train.drop_duplicates(inplace=True)
    train.dropna(inplace=True)
    train = pd.get_dummies(train, drop_first=True)
    target = train.Revenue
    train.drop("Revenue", axis=1, inplace=True)
    return train, target


def load_pulsar():
    train = pd.read_csv('data/pulsar_stars.csv')
    target = train.target_class
    train.drop('target_class', axis=1, inplace=True)
    return train, target





