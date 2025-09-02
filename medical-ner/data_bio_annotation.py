"""Модуль аннотации BIO."""

import json

from tqdm import tqdm

from data_bio_annotator import DataBIOAnnotator


def main():
    """Тело программы аннотации."""
    model_name = 'saved_model_\\data_bio_7_wo_Lizilnop_Spironol_RuBioBERT_10'
    raw_data_path = 'content\\rlsnet_texts.json'
    labeled_data_path = 'content\\labeled_data.json'

    annotator = DataBIOAnnotator(model_name)

    with open(raw_data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    labeled_data = []

    for drug in tqdm(data, ncols=80):
        drug_name = list(drug.keys())[0]
        labeled_text = {drug_name: []}
        if drug[drug_name]:
            for text in drug[drug_name]:
                labeled_text[drug_name].append(annotator.data_annotation(text))
        labeled_data.append(labeled_text)

    with open(labeled_data_path, 'w', encoding='utf-8') as file:
        json.dump(labeled_data, file, indent=4, ensure_ascii=False)
    print('Данные размечкны успешно!')


if __name__ == '__main__':
    main()
