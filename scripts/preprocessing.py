import pandas as pd
import re
from pymorphy2 import MorphAnalyzer

filename = ""
df = pd.read_csv(filename)

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

lemmas = list(map(clean_text, df['text']), total=len(df))
df['lemmas'] = lemmas
df['lemmas'] = df.lemmas.apply(lambda x: " ".join(x))

df["f1/f2"] = df["f1"]/df["f2"]
df["f7/f8"] = df["f7"]/df["f8"]
df["f1/f7"] = df["f1"]/df["f7"]
df["f4/f7"] = df["f4"]/df["f7"]
df["f3*f6"] = df["f3"]*df["f6"]

df.to_csv('data.csv', index=True)