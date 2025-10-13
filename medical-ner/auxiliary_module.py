"""Вспомогательный модуль."""

import json
import pandas as pd


TOKENS = 'tokens'
TAGS = 'tags'
OUT = 'O'


def joining_tokens(tokes):
    """Соединение токенов в текст."""
    text = ' '.join(tokes)
    return text


def load_json_data(filename):
    """Загрузка данных из JSON."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_data(filename, data):
    """Функция сохранения данных в json-файл."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def convert_to_dataframe(data):
    """Преобразование данных в pandas DataFrame."""
    rows = []
    for entry in data:
        tokens = entry[TOKENS]
        tags = entry[TAGS]
        rows.append({TOKENS: tokens, TAGS: tags})
    return pd.DataFrame(rows)


def merge_tokens_for_bert(tokens, labels):
    """
    Функция для объединения подслов в исходные слова и сохранения меток.

    tokens: Список токенов (токенизированные слова).
    labels: Список меток для каждого токена.

    Возвращает список слов с соответствующими метками.
    """
    merged_tokens = []
    merged_labels = []

    current_token = ""
    current_label = labels[0]

    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            # Убираем префикс '##' и добавляем к текущему слову
            current_token += token[2:]
        else:
            # Если есть собранное слово, добавляем его в результат
            if current_token:
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            # Начинаем новое слово
            current_token = token
            current_label = label

    # Добавляем последний токен
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels


def merge_tokens_for_t5(tokens, labels):
    """
    Функция для объединения подслов в исходные слова и сохранения меток.

    tokens: Список токенов (токенизированные слова).
    labels: Список меток для каждого токена.

    Возвращает список слов с соответствующими метками.
    """
    merged_tokens = []
    merged_labels = []

    current_token = ""
    current_label = labels[0]

    for token, label in zip(tokens, labels):
        # T5 использует символ "_" для обозначения начала нового слова
        if token.startswith("_"):
            # Если есть собранное слово, добавляем его в результат
            if current_token:
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            # Убираем символ "_" и начинаем новое слово
            current_token = token[1:]
            current_label = label
        else:
            # Добавляем подслово к текущему слову
            current_token += token

    # Добавляем последний токен
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels


def assembly_of_phrases(tokens, tags):
    """Сборка NER-словосочетаний."""
    current_token = ''
    phrases = []
    for token, tag in zip(tokens, tags):
        if tag != OUT:
            current_token += token + ' '
        elif current_token:
            phrases.append(current_token.strip())
            current_token = ''
    return phrases
