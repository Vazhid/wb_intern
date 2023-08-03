import argparse
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import re
from pymorphy2 import MorphAnalyzer

parser = argparse.ArgumentParser(description='Filename')
parser.add_argument('filename', type=str, help='Name of the data file')
args = parser.parse_args()

df = pd.read_csv(args.filename)

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

df = df[["f1/f2", "lemmas", "text", "f7/f8", "f6", "f3*f6", "f4/f7", "f1/f7", "f3"]]

model = CatBoostClassifier()
model.load_model('../models/model')

pred = model.predict(df)

np.savetxt("../results/results.txt", pred, newline="\n", fmt="%d")