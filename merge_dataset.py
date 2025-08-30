"""Модуль слтяни дадасетов."""

import json


DATASET_PATH_1 = 'content/data_bio_4.1_v2.json'
DATASET_PATH_2 = 'content/data_bio_6.json'
NEW_DATASET = 'content/data_bio_new.json'

TAGS = 'tags'
MECHANISM = 'mechanism'
ACTION = 'action'
OUT = 'O'


def load_dataset(path):
    """Функция загрузки датасетов."""
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def transformate(dataset):
    """Метод изменения тегов."""
    for sentence in dataset:
        new_tags = []
        for tag in sentence[TAGS]:
            if ACTION in tag:
                tag = tag.replace(ACTION, MECHANISM)
            elif OUT != tag:
                tag = 'O'
            new_tags.append(tag)
        sentence[TAGS] = new_tags
    return dataset


dataset_1 = load_dataset(DATASET_PATH_1)

dataset_1 = transformate(dataset_1)

dataset_2 = load_dataset(DATASET_PATH_2)
print(len(dataset_1))
print(len(dataset_2))
dataset_2.extend(dataset_1)
# print(len(dataset_2))
# Преобразуем словари в JSON-строки
unique_sentence_entities: set[str] = {json.dumps(entry,
                                                 sort_keys=True,
                                                 ensure_ascii=False)
                                      for entry in dataset_2}
# print(len(unique_sentence_entities))
# обратное преобразование
unique_sentence_entities: list[dict] = [json.loads(item)
                                        for item in unique_sentence_entities]
# print(len(unique_sentence_entities))
with open(NEW_DATASET, 'w', encoding='utf-8') as file:
    json.dump(unique_sentence_entities, file, indent=4, ensure_ascii=False)

print('Датасеты слиты успешно!')
