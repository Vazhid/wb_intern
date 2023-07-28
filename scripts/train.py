import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool
from catboost.utils import get_roc_curve, select_threshold

filename = ""
df = pd.read_csv(filename)

X = df.drop(["label"])
y = df["label"]

pool = Pool(
    X, y,
    text_features=["lemmas", "text"]
)

model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.01,
                           loss_function='Logloss').fit(pool, verbose=False)

roc_curve_values = get_roc_curve(model, pool)
boundary = select_threshold(model, curve=roc_curve_values)
model.set_probability_threshold(boundary)