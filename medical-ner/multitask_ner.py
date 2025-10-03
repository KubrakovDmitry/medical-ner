"""Модуль обучения модели NER нлтации BIO."""

import json

import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer,
                          BertForTokenClassification,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback,
                          AutoConfig,
                          BertPreTrainedModel,
                          BertModel)
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from torch import nn

TOKENS = 'tokens'
TAGS = 'tags'
MAX_LENGTH = 'max_length'
OUT_TAG = 'O'

logged_metrics = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json_data(filename):
    """Функция загрузки данных из json-файла."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_to_dataframe(data):
    """Конвертация в датафрейм pandas."""
    rows = []
    for entry in data:
        tokens = entry[TOKENS]
        tags = entry[TAGS]
        rows.append({TOKENS: tokens, TAGS: tags})
    return pd.DataFrame(rows)


def tokenize_and_align_labels(examples):
    """Токенизация и выравнивание."""
    global max_length
    tokenized_inputs = tokenizer(examples[TOKENS],
                                 truncation=True,
                                 is_split_into_words=True,
                                 padding=MAX_LENGTH,
                                 max_length=max_length)
    labels = []
    for i, label in enumerate(examples[TAGS]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id.get(label[word_idx],
                                              label2id[OUT_TAG]))
        if tokenizer.sep_token_id:
            sep_index = tokenized_inputs['input_ids'][i].index(
                tokenizer.sep_token_id)
            label_ids[sep_index] = -100
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    tokenized_inputs['sentence_labels'] = examples['sentence_labels']
    return tokenized_inputs


def compute_metrics(p):
    """Подсчёт метрик."""
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
    metrics_to_log = {}
    print(results)
    for label, metric_values in results.items():
        if isinstance(metric_values, dict):
            for metric_name, value in metric_values.items():
                metrics_to_log[f"{label}_{metric_name}"] = value
        else:
            metrics_to_log[label] = metric_values
    logged_metrics.append(metrics_to_log)
    return metrics_to_log


def add_sentence_labels(example):
    """Добавление поля метки наличия сущностей в предложении."""
    example["sentnce_labels"] = int(any(tag != OUT_TAG
                                        for tag in example[TAGS]))
    return example


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
        self.classifier = nn.Linear(config.hidden_size, num_labels)

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
        if labels is not None and sentence_labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = token_logits.view(-1, self.num_labels)
            active_labels = labels.view(-1)[active_loss]
            loss_token = self.loss_fct_token(active_logits, active_labels)

            loss_sentence = self.loss_fct_token(active_logits, active_labels)

            loss = loss_token + self.lambda_bin * loss_sentence

        return {
            "loss": loss,
            "logits": token_logits,
            "sentence_logits": sentence_logits
        }


annotation = 'content//data_bio_8_1.json'
model_name = 'alexyalunin/RuBioBERT'
EPOCH = 10

FILTER_TAGS = ['O']

data = load_json_data(annotation)

filtered_data = []
o_counter = 0
for item in data:
    if all(tag in FILTER_TAGS for tag in item[TAGS]):
        if o_counter <= 150:
            o_counter += 1
        else:
            continue
    filtered_data.append(item)


df = convert_to_dataframe(filtered_data)
train_df, test_df = train_test_split(df, test_size=0.15)

label_list = set()
for i in range(df.shape[0]):
    label_list.update(list(df.tags.iloc[i]))

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
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

metric = evaluate.load("seqeval")

STOP_EPOCH = 3
WARMUP_RADIO = 0.1
early_stopping = EarlyStoppingCallback(early_stopping_patience=STOP_EPOCH)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=WARMUP_RADIO,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCH,
    weight_decay=0.01
)

for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()
print(f'Обучение модели {model_name} с {EPOCH} эпох')

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()


def convert_to_serializable(obj):
    """Конвертация сложные структуры в простые."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


for epoch, metrics in enumerate(logged_metrics, 1):
    metrics["epoch"] = epoch

logged_metrics = convert_to_serializable(logged_metrics)

with open("training_metrics.json", "w", encoding="utf-8") as f:
    json.dump(logged_metrics, f, indent=4, ensure_ascii=False)

annotation_name = annotation.split('.')[0]
model.save_pretrained('./saved_model' + '_' + annotation_name.replace('content/', '') + '_' +
                      model_name.split('/')[-1] + '_' + str(EPOCH))
tokenizer.save_pretrained('./saved_model' + '_' + annotation_name.replace('content/', '') + '_' +
                          model_name.split('/')[-1] + '_' + str(EPOCH))
results = trainer.evaluate()
print(results)

print('Работа программы успешно завершена!')
