"""Модуль аннотатора BIO."""

from transformers import (AutoModelForTokenClassification,
                          AutoTokenizer)
import torch


class DataBIOAnnotator:
    """Разметчик данных по нотациии BIO."""

    __TOKENS = 'tokens'
    __TAGS = 'tags'

    def __init__(self, model_name):
        """Коструктор аннотатора."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            trust_remote_code=True)

    @classmethod
    def merge_tokens(cls, tokens, labels):
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
            # Убираем префикс '##' и добавляем к текущему слову
            if token.startswith("##"):
                current_token += token[2:]
            else:
                # Если есть собранное слово, добавляем его в результат
                if current_token:
                    merged_tokens.append(current_token)
                    merged_labels.append(current_label)
                current_token = token  # Начинаем новое слово
                current_label = label
        # Добавляем последний токен
        if current_token:
            merged_tokens.append(current_token)
            merged_labels.append(current_label)
        return merged_tokens, merged_labels

    def data_annotation(self, text):
        """Метод разметки BIO."""
        token_clear = []
        tag_clear = []

        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding=True)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [self.model.config.id2label[t.item()]
                                 for t in predictions[0]]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens, predicted_token_class = self.merge_tokens(
            tokens,
            predicted_token_class)

        for token, tag in zip(tokens, predicted_token_class):
            if token == '[SEP]' or token == '[CLS]':
                continue
            token_clear.append(token)
            tag_clear.append(tag)

        return {self.__TOKENS: token_clear,
                self.__TAGS: tag_clear}
