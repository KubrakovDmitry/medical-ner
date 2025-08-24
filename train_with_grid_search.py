import logging
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import seaborn as sns
import matplotlib.pyplot as plt


# Настройка логирования
logging.basicConfig(filename='training_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Загрузка данных из JSON
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Сохранение данных в JSON
def save_json_data(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file)


# Преобразование данных в pandas DataFrame
def convert_to_dataframe(data):
    rows = []
    for entry in data:
        tokens = entry['tokens']
        tags = entry['tags']
        rows.append({'tokens': tokens, 'tags': tags})
    return pd.DataFrame(rows)


# Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(examples):
    global max_length
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=max_length)
    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignored token
            else:
                label_ids.append(label2id.get(label[word_idx], label2id['O']))
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# Функция для расчёта метрик
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
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


# Задание параметров для Grid Search
annotations = ['content/data_bio_8_1.json']
model_names = [
                'alexyalunin/RuBioBERT',
               ]
dataset_path = 'content\\train.csv'
test_path = 'content\\test.json'

# Сетка Grid
param_grid = {
    #"learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "train_batch_size": [2, 4, 8, 16],
    "test_batch_size": [2, 4, 8, 16],
    #"num_train_epochs": list(range(3, 7))
}

# Создание всех комбинаций параметров
grid = list(ParameterGrid(param_grid))

results_list = []

for annotation in annotations:
    data = load_json_data(annotation)
    df_1 = convert_to_dataframe(data)
    data, _ = train_test_split(data, test_size=0.1)
    save_json_data(_, test_path)
    df_2 = convert_to_dataframe(data)

    # Разделение данных на обучающий и тестовый наборы
    train_df, test_df = train_test_split(df_2, test_size=0.1)
    label_list = list(set(tag for tags in df_1["tags"] for tag in tags))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    for model_name in model_names:
        # Загрузка токенизатора и модели
        if 't5' in model_name.lower():
            max_length = 512
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), trust_remote_code=True)
        elif 'bert' in model_name.lower():
            max_length = 512
            tokenizer = BertTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
            model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), trust_remote_code=True)

        model.config.label2id = label2id
        model.config.id2label = id2label

        tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        metric = load_metric("seqeval")

        for params in grid:
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=params['train_batch_size'],
                per_device_eval_batch_size=params['test_batch_size'],
                num_train_epochs=4,
                weight_decay=0.01,
                logging_dir="logs",
                logging_steps=10,
                log_level="info",
                save_strategy="no"
            )

            for param in model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            print(f'обучение модели {model_name}  с {6} эпох')

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"]
            )

            trainer.train()
            # Сохранение модели
            annotation_name = annotation.split('.')[0]
            model.save_pretrained('models\\saved_model' + '_' + annotation_name.replace('content/', '') + '_' + model_name.split('/')[-1] + '_' + str(params['train_batch_size']) + '_' + str(params['test_batch_size']))
            tokenizer.save_pretrained('models\\saved_model' + '_' + annotation_name.replace('content/', '') + '_' + model_name.split('/')[-1] + '_' + str(params['train_batch_size']) + '_' + str(params['test_batch_size']))
            result = trainer.evaluate()
            results_list.append({
                "model_name": model_name,
                "train_batch_size": params["train_batch_size"],
                "test_batch_size": params["test_batch_size"],
                "eval_precision": result["eval_precision"],
                "eval_recall": result["eval_recall"],
                "eval_f1": result["eval_f1"],
                "eval_accuracy": result["eval_accuracy"]
            })

# Сохранение результатов
results_df = pd.DataFrame(results_list)
results_df.to_csv("grid_search_results.csv", index=False)

# Визуализация
# Преобразование данных для тепловой карты
pivot_columns = ["train_batch_size", "test_batch_size"]
metrics = ["eval_f1", "eval_precision", "eval_recall", "eval_accuracy"]

for metric_name in metrics:
    plt.figure(figsize=(12, 8))
    pivot_table = results_df.pivot_table(
        values=metric_name, 
        index="train_batch_size", 
        columns="test_batch_size", 
        aggfunc="mean"
    )
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title(f"Grid Search Results: {metric_name.replace('eval_', '').capitalize()} Heatmap")
    plt.xlabel("Test Batch Size")
    plt.ylabel("Train Batch Size")
    plt.tight_layout()
    plt.savefig(f"images/grid_search/mark_sent_2_rubiobert_batches_{metric_name}.png", dpi=300)
    # plt.show()

print("Работа программы успешно завершена!")
