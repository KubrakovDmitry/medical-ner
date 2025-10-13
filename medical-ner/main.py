"""FastAPI - приложение для NER."""

import json

from fastapi import FastAPI, Body, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ner_model import NERModel
from ner_trainer import NERTrainer


TEXT = "text"
ANNOTATION = 'dataset'
MODEL_NAME = 'model_name'
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
async def get_model_page():
    """Переход на страницу обучения модели."""
    return FileResponse('train_page.html')


@app.post('/train-ner-model')
async def train(
    dataset: UploadFile = File(...),
    model_name: str = Form(...),
    epochs: int = Form(3),
    train_batch_size: int = Form(4),
    eval_batch_size: int = Form(4),
    learning_rate: float = Form(1.5e-5),
    decay: float = Form(0.01),
    log: int = Form(30),
    lr_type: str = Form(""),
    warmup_radio: float = Form(0.05),
    fp16: bool = Form(False),
    grad_accum: int = Form(2)
):
    """Обучение модели для NER."""
    contents = await dataset.read()
    annotation = json.loads(contents)

    results = trainer.train(
        dataset_name=dataset.filename,
        annotation=annotation,
        model_name=model_name,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        decay=decay,
        logging_step=log,
        lr_scheduler_type=lr_type,
        warmup_ratio=warmup_radio,
        fp16=fp16,
        gradient_accumulation_steps=grad_accum
    )
    return {
        "message": "Обучение завершено успешно!",
        "metrics": results["metrics"]
    }
