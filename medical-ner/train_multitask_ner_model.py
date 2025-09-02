"""Модуль обучения """

import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import (TrainingArguments, Trainer, AutoConfig,
                          BertModel, BertPreTrainedModel, AutoTokenizer)
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from auxiliary_module import (load_json_data, save_json_data,
                              convert_to_dataframe)


metric = load_metric("seqeval")

TOKEN = 'tokens'
TAGS = 'tags'

TRAIN = 'train'
TEST = 'test'
OUT_TAG = 'O'
SENTENCE_LABELS = 'sentence_labels'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Настройка логирования
logging.basicConfig(filename='training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def out_filter(data):
    """Фильтр для OUT."""
    return [sent for sent in data
            if any(tag != OUT_TAG for tag in sent[TAGS])]


def add_sentence_labels(example):
    """Добавление поля метки наличия сущностей в предложении."""
    example["sentence_labels"] = int(any(tag != OUT_TAG
                                        for tag in example[TAGS]))
    return example


# Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(examples):
    global max_length
    tokenized_inputs = tokenizer(examples[TOKEN],
                                 truncation=True,
                                 is_split_into_words=True,
                                 padding='max_length',
                                 max_length=max_length)
    labels = []
    for i, label in enumerate(examples[TAGS]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignored token
            else:
                label_ids.append(label2id.get(label[word_idx],
                                              label2id[OUT_TAG]))
        # Пометить последний токен [SEP] как -100
        if tokenizer.sep_token_id:
            sep_index = tokenized_inputs['input_ids'][i].index(
                tokenizer.sep_token_id)
            label_ids[sep_index] = -100
        
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    tokenized_inputs[SENTENCE_LABELS] = examples[SENTENCE_LABELS]
    return tokenized_inputs


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred

#     if isinstance(predictions, tuple) or isinstance(predictions, list):
#         predictions = predictions[0]

#     predictions = np.argmax(predictions, axis=2)

#     true_labels = []
#     pred_labels = []

#     for prediction, label in zip(predictions, labels):
#         true_label = []
#         pred_label = []
#         for p, l in zip(prediction, label):
#             if l != -100:
#                 true_label.append(id2label[l])
#                 pred_label.append(id2label[p])
#         true_labels.append(true_label)
#         pred_labels.append(pred_label)

#     results = metric.compute(predictions=pred_labels, references=true_labels)
#     print("Результаты метрики:", results)
#     flat_results = {}
#     for k, v in results.items():
#         if isinstance(v, dict):
#             for subk, subv in v.items():
#                 flat_results[f"{k}_{subk}"] = subv
#         else:
#             flat_results[k] = v
#     return flat_results


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Разделяем логи для токенов и предложений
    token_logits, sentence_logits = predictions
    token_labels, sentence_labels = labels

    # --- NER часть ---
    token_preds = np.argmax(token_logits, axis=-1)

    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(token_preds, token_labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:  # игнорируем паддинг
                true_labels.append(id2label[l])
                pred_labels.append(id2label[p])

    ner_results = metric.compute(predictions=[pred_labels], references=[true_labels])

    # --- Sentence классификация ---
    sentence_preds = np.argmax(sentence_logits, axis=-1)
    sentence_accuracy = (sentence_preds == sentence_labels).mean()

    return {
        "ner_precision": ner_results["overall_precision"],
        "ner_recall": ner_results["overall_recall"],
        "ner_f1": ner_results["overall_f1"],
        "ner_accuracy": ner_results["overall_accuracy"],
        "sentence_accuracy": sentence_accuracy,
    }


class BertForTokenAndSentenceClassification(BertPreTrainedModel):
    """
    Многозадачная модель.

    Модель решает задачи классификации предложений
    и токенов NER.
    """

    def __init__(self, config, *, num_labels=None, lambda_bin=0.3):
        """Конструктор."""
        super().__init__(config)
        self.num_labels = (num_labels if num_labels is not None
                           else config.num_labels)
        self.lambda_bin = lambda_bin

        self.bert = BertModel(config, add_pooling_layer=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.sentence_classifier = nn.Linear(config.hidden_size, 2)

        self.loss_fct_token = nn.CrossEntropyLoss()
        self.loss_fct_sentence = nn.CrossEntropyLoss()

        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                sentence_labels=None,
                **kwargs):
        """Обучение."""
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        sequence_output = self.dropout(sequence_output)
        token_logits = self.classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        sentence_logits = self.sentence_classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_token = self.loss_fct_token(
                token_logits.view(-1, self.num_labels),
                labels.view(-1))

            loss_sentence = self.loss_fct_sentence(sentence_logits,
                                                   sentence_labels)

            loss = loss_token + self.lambda_bin * loss_sentence

        return {
            "loss": loss,
            "logits": token_logits,
            "sentence_logits": sentence_logits
        }


annotation = 'content\\data_bio_8_1.json'
model_name = 'alexyalunin/RuBioBERT'
dataset_path = 'content\\train.csv'
test_path = 'content\\test.json'

EPOCHS = 8
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 1.5e-5
DECAY = 0.01
LOGGING_STEP = 30
LR_SCHEDULER_TYPE = "linear"
WARMUP_RADIO = 0.06
FP16 = True
GRADIENT_ACCUMULATION_STEP = 2

data = out_filter(load_json_data(annotation))
df_1 = convert_to_dataframe(data)
data, _ = train_test_split(data, test_size=0.1)
save_json_data(test_path, _)
df_2 = convert_to_dataframe(data)
# Разделение данных на обучающий и тестовый наборы
train_df, test_df = train_test_split(df_2, test_size=0.1)
label_list = list()
for i in range(df_1.shape[0]):
    label_list.extend((list(df_1.tags.iloc[i])))
label_list = set(label_list)
# Пример меток
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
# Конвертация в формат Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({
    TRAIN: train_dataset,
    TEST: test_dataset
})

max_length = 512
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,)
model = BertForTokenAndSentenceClassification.from_pretrained(
    model_name,
    config=config,
    num_labels=len(label_list),
    lambda_bin=0.3
    ).to(device)

model.config.label2id = label2id
model.config.id2label = id2label

dataset_dict = dataset_dict.map(add_sentence_labels)
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Обновленная функция для вычисления метрик
# Настройки для тренировки
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=DECAY,
    logging_dir="logs",             # Директория для логов
    logging_steps=LOGGING_STEP,               # Частота логирования
    log_level="info",               # Уровень логирования
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
    warmup_ratio=WARMUP_RADIO,
    fp16=FP16,
    lr_scheduler_type=LR_SCHEDULER_TYPE
)


for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_datasets[TRAIN],
    eval_dataset=tokenized_datasets[TEST]
)

trainer.train()

# Сохранение модели
annotation_name = annotation.split('.')[0]
model_name = ('models/model' + '_'
    + annotation_name.replace('content\\', '')
    + '_' + model_name.split('/')[-1]
    + '_' + str(datetime.now().strftime("%Y%m%d_%H%M%S")))
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

# Оценка модели
results = trainer.evaluate()
print(results)

for entry in trainer.state.log_history:
    print(entry.keys())

metrics = pd.DataFrame(
    [entry for entry in trainer.state.log_history if "eval_loss" in entry])
metrics = metrics[[col for col in ["epoch", "eval_ner_precision",
                                   "eval_ner_recall", "eval_ner_f1",
                                   "eval_loss"]
                   if col in metrics.columns]]
print(metrics.columns)
print(f'metrics = {metrics}')
metrics.to_csv('метрики модели по эпохам.csv', index=False)
plt.plot(metrics["epoch"], metrics["eval_loss"], label="Loss")
plt.plot(metrics["epoch"], metrics["eval_ner_precision"],
         label="Precision")
plt.plot(metrics["epoch"], metrics["eval_ner_recall"], label="Recall")
plt.plot(metrics["epoch"], metrics["eval_ner_f1"], label="F1")

plt.xlabel("Эпохи")
plt.ylabel("Значения метрик")
plt.legend()
plt.title("Метрики по эпохам")
plt.savefig(f'images\\{model_name.replace("models/", "")}.png', dpi=300, bbox_inches="tight")

print('Работа программы успешно завершина!')