import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch


# Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples['left'],
                          examples['right'],
                          padding=True,
                          truncation=True,
                          max_length=max_length)
    tokenized['labels'] = examples['type']
    return tokenized


def compute_metrics(p):
    """Функция для расчёта метрик."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    pred_labels = [id2label[pred] for pred in predictions]
    true_labels = [id2label[true] for true in labels]
    # Вычисляем метрики
    precision = precision_score(true_labels, pred_labels, average='binary')
    recall = recall_score(true_labels, pred_labels, average='binary')
    f1 = f1_score(true_labels, pred_labels, average='binary')
    accuracy = accuracy_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


# Задание параметров для Grid Search
annotations = 'content\\changed_data_relations_3.csv'
model_names = [
                'DeepPavlov/rubert-base-cased',
                'alexyalunin/RuBioBERT',
              ]
dataset_path = 'content\\train_relations_3.csv'
test_path = 'content\\test_relations_3.csv'
df = pd.read_csv(annotations,
                 delimiter='\t',
                 encoding='utf-8'
                 )

start = 3
finish = 11
# # Укажите количество фолдов

fold_results = []
df['left'] =  df[['from', 'sent_from']].agg(' '.join, axis=1)
df['right'] = df[['to', 'sent_to']].agg(' '.join, axis=1)
new_df = df[['left', 'right', 'type']]
new_df['type'] = new_df['type'].map({'link': 1,
                                     'Not_link': 0,
                                     'sub_group': 1,
                                     'combination': 1}).astype(object)
train_df, test_df = train_test_split(new_df, test_size=0.1)
new_df.to_csv('content\\new_relations_3.csv', sep='\t', index=False)
test_df.to_csv('content\\test_relations_3.csv', sep='\t', index=False)
train_df, test_df = train_test_split(train_df, test_size=0.1)
print(f'метки train = {train_df["type"].unique()}')
print(f'метки test = {test_df["type"].unique()}')
label_list = list()
for i in range(new_df.shape[0]):
    label_list.append((new_df.type.iloc[i]))
label_list = set(label_list)
label2id = {0: 0, 1: 1}
id2label = {0: 0, 1: 1}
print('len(label_list) =', len(label_list))
# Конвертация в формат Dataset
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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_list), trust_remote_code=True)
    elif 'bert' in model_name.lower():
        max_length = 512
        tokenizer = BertTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_list), trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)
    model.to(device)
    model.config.label2id = label2id
    model.config.id2label = id2label
    tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    # Обновленная функция для вычисления метрик
    for num_epoch in range(start, finish):
        # Настройки для тренировки
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_epoch,
            weight_decay=0.1
        )
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        print(f'обучение модели {model_name}  с {num_epoch} эпох')
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            # compute_loss=compute_loss,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test']
        )
        trainer.train()
        # Сохранение модели
        annotation_name = annotations.split('.')[0]
        model.save_pretrained('models/saved_model' + '_' + annotation_name.replace('content/', '') + '_' + model_name.split('/')[-1] + '_' + str(num_epoch))
        tokenizer.save_pretrained('models/saved_model' + '_' + annotation_name.replace('content/', '') + '_' + model_name.split('/')[-1] + '_' + str(num_epoch))
        # Оценка модели
        results = trainer.evaluate()
        print(results)
print('Обучение моделей классификации связей успешно завершина!')