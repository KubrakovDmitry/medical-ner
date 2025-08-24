import json
import pandas as pd


def joining_tokens(tokes):
    text = ' '.join(tokes)
    return text


# Загрузка данных из JSON
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_data(filename, data):
    """Функция сохранения данных в json-файл."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# Преобразование данных в pandas DataFrame
def convert_to_dataframe(data):
    rows = []
    for entry in data:
        tokens = entry['words']
        tags = entry['tags']
        rows.append({'words': tokens, 'tags': tags})
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
            current_token += token[2:]  # Убираем префикс '##' и добавляем к текущему слову
        else:
            if current_token:  # Если есть собранное слово, добавляем его в результат
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token  # Начинаем новое слово
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
        if token.startswith("_"):  # T5 использует символ "_" для обозначения начала нового слова
            if current_token:  # Если есть собранное слово, добавляем его в результат
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token[1:]  # Убираем символ "_" и начинаем новое слово
            current_label = label
        else:
            current_token += token  # Добавляем подслово к текущему слову
    
    # Добавляем последний токен
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)
    
    return merged_tokens, merged_labels