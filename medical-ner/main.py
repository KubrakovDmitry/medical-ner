"""FastAPI - приложение для NER."""

import json

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ner_model import NERModel
from ner_trainer import NERTrainer


TEXT = "text"
ANNOTATION = 'dataset'
MODEL_NAME = 'name_model'
EPOCHS = 'epochs'
TRAIN_BATCH_SIZE = 'train_batch_size'
EVAL_BATCH_SIZE = 'eval_batch_size'
LEARNING_RATE = 'learning_rate'
DECAY = 'decay'
LOGGING_STEP = 'log'
LR_SCHEDULER_TYPE = 'lr_type'
WARMUP_RATIO = 'warmup_radio'
FP16 = 'fp16'
GRADIENT_ACCUMULATION_STEPS = 'grad_accum'

app = FastAPI()
model = NERModel()
trainer = NERTrainer()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    """Вход на сайт."""
    return FileResponse("index.html")


@app.post("/ner")
def ner(data=Body()):
    """Поиск NER во входном тексте."""
    entities = model.predict(data[TEXT])
    text = ''

    for i, entity in enumerate(entities, 1):
        text += f'{i}) {entity};\n'

    return {TEXT: text[:text.rfind(';\n')] + '.'}


@app.get('/model')
def get_model_page():
    """Переход на страницу обучения модели."""
    return FileResponse('train_page.html')


@app.post('/train-ner-model')
def train(data=Body()):
    """Обучение модели для NER."""

    trainer.train(
        annotation=None,
        model_name=data[MODEL_NAME],
        epochs=data[EPOCHS],
        train_batch_size=data[TRAIN_BATCH_SIZE],
        eval_batch_size=data[EVAL_BATCH_SIZE],
        learning_rate=data[LEARNING_RATE],
        decay=data[DECAY],
        logging_step=data[LOGGING_STEP],
        lr_scheduler_type=data[LR_SCHEDULER_TYPE],
        warmup_ratio=data[WARMUP_RATIO],
        fp16=data[FP16],
        gradient_accumulation_steps=data[GRADIENT_ACCUMULATION_STEPS]
    )
    return {'message': 'Обучение модели NER завершено успешно!'}
