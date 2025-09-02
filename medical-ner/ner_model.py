"""Программа для тестирования языковых моделей."""

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

from auxiliary_module import (load_json_data, merge_tokens_for_bert,
                              convert_to_dataframe, assembly_of_phrases)


class NERModel:
    """Класс, реализующий NER."""

    __model_name = 'models\\model_data_bio_8_1_RuBioBERT_20250819_191511'
    __annotation = 'data_bio_8_1.json'

    def __init__(self, model_name=None):
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.__model_name

        data = load_json_data(self.__annotation)
        df = convert_to_dataframe(data)

        label_list = list()
        for i in range(df.shape[0]):
            label_list.extend((list(df.tags.iloc[i])))

        label_list = set(label_list)

        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(label_list),
            trust_remote_code=True)

    def predict(self, text):
        """Метод выполнения NER."""

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [self.model.config.id2label[t.item()]
                                 for t in predictions[0]]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens, predicted_token_class = merge_tokens_for_bert(
            tokens,
            predicted_token_class)
        return assembly_of_phrases(tokens, predicted_token_class)


def main():
    """Точка входа в программу."""
    print('Работа выделителя NER началась!')

    text = (
        'Периндоприл оказывает сосудорасширяющее действие , '
        'способствует восстановлению эластичности крупных артерий и структуры сосудистой '
        'стенки мелких артерий , а также уменьшает гипертрофию левого желудочка . '
        'Одновременное применение тиазидных диуретиков group усиливает выраженность '
        'антигипертензивного эффекта. Сердечная недостаточность Периндоприл prepare '
        'нормализует работу сердца , снижая преднагрузку и постнагрузку. '
        'У пациентов с хронической сердечной недостаточностью (ХСН) , '
        'получавших периндоприл , было выявлено снижение давления наполнения в левом '
        'и правом желудочках сердца, снижение ОПСС, повышение сердечного выброса '
        'и увеличение сердечного индекса .')

    model = NERModel()
    result = model.predict(text)

    print(result)
    print('Работа программы успешно завершина!')


if __name__ == '__main__':
    main()
