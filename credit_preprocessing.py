import pandas as pd



def load_credit(tukey_multiplier=3.5):
    train = pd.read_csv('data/cs-training.csv',index_col=0)
    rows = train.shape[0]
    incomeq1 = train.MonthlyIncome.quantile(0.25)
    incomeq3 = train.MonthlyIncome.quantile(0.75)
    iqr = incomeq3 - incomeq1

    outlier_min = incomeq1 - (iqr) * 1.5
    outlier_max = incomeq3 + (iqr) * 1.5

    train = train[(train.MonthlyIncome > outlier_min) & (train.MonthlyIncome < outlier_max)]
    pruned_rows = train.shape[0] - rows
    print(f"{pruned_rows} were removed as outliers")
    
    target = train['SeriousDlqin2yrs']
    train['NumberOfDependents'].fillna(0)
    train['MonthlyIncome'].fillna(train.MonthlyIncome.median())
    train.drop('SeriousDlqin2yrs', axis=1, inplace=True)
    return train, target
    
