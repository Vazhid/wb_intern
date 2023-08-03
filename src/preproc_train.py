import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
from catboost import CatBoostClassifier
from catboost import Pool
from catboost.utils import get_roc_curve, select_threshold

df = pd.read_csv("../data/wb_school_task_2.csv")

m = MorphAnalyzer()
regex = re.compile("[А-Яа-яA-z]+")

def words_only(text, regex=regex):
    try:
        return regex.findall(text.lower())
    except:
        return []

def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form

def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]

def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    return lemmas

lemmas = list(map(clean_text, df['text']))
df['lemmas'] = lemmas
df['lemmas'] = df.lemmas.apply(lambda x: " ".join(x))

df["f1/f2"] = df["f1"]/df["f2"]
df["f7/f8"] = df["f7"]/df["f8"]
df["f1/f7"] = df["f1"]/df["f7"]
df["f4/f7"] = df["f4"]/df["f7"]
df["f3*f6"] = df["f3"]*df["f6"]

df = df[["f1/f2", "lemmas", "text", "f7/f8", "f6", "f3*f6", "f4/f7", "f1/f7", "f3", "label"]]

X = df.drop(["label"])
y = df["label"]

pool = Pool(
    X, y,
    text_features=["lemmas", "text"]
)

model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.01,
                           loss_function='Logloss').fit(pool, verbose=100)

roc_curve_values = get_roc_curve(model, pool)
boundary = select_threshold(model, curve=roc_curve_values)
model.set_probability_threshold(boundary)

model.save_model("../model/model_1")