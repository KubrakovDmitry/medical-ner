"""Модуль обучения """

import logging
import json
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (BertTokenizerFast, BertForTokenClassification,
                          TrainingArguments, Trainer)
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np
import matplotlib.pyplot as plt

from auxiliary_module import (load_json_data, save_json_data,
                              convert_to_dataframe)


metric = load_metric("seqeval")

TOKEN = 'tokens'
TAGS = 'tags'

TRAIN = 'train'
TEST = 'test'
OUT_TAG = 'O'
# Настройка логирования
logging.basicConfig(filename='training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def out_filter(data):
    """Фильтр для OUT."""
    return [sent for sent in data
            if any(tag != OUT_TAG for tag in sent[TAGS])]


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
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        pred_label = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_label.append(id2label[l])
                pred_label.append(id2label[p])
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    results = metric.compute(predictions=pred_labels, references=true_labels)
    print("Результаты метрики:", results)
    flat_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat_results[f"{k}_{subk}"] = subv
        else:
            flat_results[k] = v
    return flat_results


annotation = 'content\\data_bio_4.json'
# DeepPavlov/rubert-base-cased
# alexyalunin/RuBioBERT
model_name = 'DeepPavlov/rubert-base-cased'
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
tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                              trust_remote_code=True)
model = BertForTokenClassification.from_pretrained(model_name,
                                                   num_labels=len(label_list),
                                                   trust_remote_code=True)

model.config.label2id = label2id
model.config.id2label = id2label

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
metrics = metrics[[col for col in ["epoch", "eval_overall_precision",
                                   "eval_overall_recall", "eval_overall_f1",
                                   "eval_loss"]
                   if col in metrics.columns]]
print(metrics.columns)
print(f'metrics = {metrics}')
metrics.to_csv('метрики модели по эпохам.csv', index=False)
plt.plot(metrics["epoch"], metrics["eval_loss"], label="Loss")
plt.plot(metrics["epoch"], metrics["eval_overall_precision"],
         label="Precision")
plt.plot(metrics["epoch"], metrics["eval_overall_recall"], label="Recall")
plt.plot(metrics["epoch"], metrics["eval_overall_f1"], label="F1")

plt.xlabel("Эпохи")
plt.ylabel("Значения метрик")
plt.legend()
plt.title("Метрики по эпохам")
plt.savefig(f'images\\{model_name.replace("models/", "")}.png', dpi=300, bbox_inches="tight")

print('Работа программы успешно завершина!')