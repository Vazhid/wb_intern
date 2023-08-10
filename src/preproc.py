import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
import json
from pathlib import Path
import os

REGEX = re.compile("[А-Яа-яA-z]+")
MORPH_ANALYZER = MorphAnalyzer()

def words_only(text: str, regex=REGEX) -> list:
    ''' Функция получает на вход строку и возвращает список соответсвующих шаблону строк '''
    try:
        return regex.findall(text.lower())
    except:
        return []

def lemmatize_word(token: str, pymorphy=MORPH_ANALYZER) -> str:
    ''' Функция получает на вход строку(слово) и возвращает слово, приведенное к нормальной форме '''
    return pymorphy.parse(token)[0].normal_form

def clean_text(text: str) -> list:
    ''' Функция получает на вход строку(текст) и возвращает список нормализованных и очищенных слов текста '''
    tokens = words_only(text)
    lemmas = [lemmatize_word(w) for w in tokens]
    return lemmas

def main():
    os.chdir(Path(__file__).parent.parent) # изменение текущей директории на корневую директорию проекта
    with open("config/config.json", "r") as jsonfile: # считывание файла конфигурации
        config = json.load(jsonfile)

    df = pd.read_csv(config["file_for_preprocessing"]) # считывание csv-файла для обработки

    lemmas = list(map(clean_text, df['text'])) # нормализация текста отзывов и запись этих текстов в переменную lemmas
    df['lemmas'] = lemmas # добавление нового признака lemmas в датасет
    df['lemmas'] = df.lemmas.apply(lambda x: " ".join(x))

    df["f1/f2"] = df["f1"]/df["f2"] # генерация новых признаков
    df["f7/f8"] = df["f7"]/df["f8"]
    df["f1/f7"] = df["f1"]/df["f7"]
    df["f4/f7"] = df["f4"]/df["f7"]
    df["f3*f6"] = df["f3"]*df["f6"]

    df = df[["f1/f2", "lemmas", "text", "f7/f8", "f6", "f3*f6", "f4/f7", "f1/f7", "f3", "label"]] # оставим в датасете отобранные признаки

    df.to_csv("data/wb_school_task_2_preproc.csv") # сохранение датасета, который готов для обучения модели

if __name__ == '__main__':
    main()