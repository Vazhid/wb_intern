import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool
from catboost.utils import get_roc_curve, select_threshold
import json
from pathlib import Path
import os

def main():
    os.chdir(Path(__file__).parent.parent) # изменение текущей директории на корневую директорию проекта
    with open("config/config.json", "r") as jsonfile: # считывание файла конфигурации
        config = json.load(jsonfile)

    df = pd.read_csv(config["file_for_training"]) # считывание csv-файла для обучения

    X = df.drop(["label"]) # разделение датасета на признаки
    y = df["label"] # и таргет

    pool = Pool( 
        X, y,
        text_features=["lemmas", "text"]
    )

    model = CatBoostClassifier(iterations=1000,
                               learning_rate=0.01,
                               loss_function='Logloss').fit(pool, verbose=False) # обучение модели с установленными параметрами

    roc_curve_values = get_roc_curve(model, pool)
    boundary = select_threshold(model, curve=roc_curve_values) # подбор threshold
    model.set_probability_threshold(boundary) # установление подобранного threshold

    model.save_model("../model/model_1") # сохранение обученной модели

if __name__ == '__main__':
    main()