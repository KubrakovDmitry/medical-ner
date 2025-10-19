"""FastAPI - приложение для NER."""

import json
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from multiprocessing import Process, Manager, Value

from fastapi import FastAPI, Body, File, Form, UploadFile, BackgroundTasks
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

manager = None
training_progress = None
should_interrupt = None
training_process: Optional[Process] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненым циклом приложения."""
    global manager, training_progress, should_interrupt
    should_interrupt = Value('b', False)
    manager = Manager()
    training_progress = manager.dict({
        'status': 'not_started',
        'current_epoch': 0,
        'total_epochs': 0,
        'current_step': 0,
        'total_steps': 0,
        'progress': 0.0,
        'loss': None,
        'current_metrics': None
    })

    try:
        yield
    finally:
        global training_process
        if training_process and training_process.is_alive():
            print('Прекращение обучения модели NER')
            should_interrupt.value = True
            training_process.join(timeout=10)
            if training_process.is_alive():
                training_process.terminate()
        manager.shutdown()


app = FastAPI(lifespan=lifespan)
model = NERModel()
trainer = NERTrainer()

app.mount("/static", StaticFiles(directory="static"), name="static")


def _progress_callback_factory(progress_proxy):
    """
    Возвращение функцию, которую Trainer будет вызывать
    (в дочернем процессе).
    """
    def progress_cb(status: str, **kwargs):
        """Обновление прогресса обучение."""
        progress_proxy[status] = status
        for k, v in kwargs.items():
            progress_proxy[k] = v
    return progress_cb


def run_training_process(dataset_name: str,
                         annotation,
                         model_name: str,
                         epochs: int,
                         train_batch_size: int,
                         eval_batch_size: int,
                         learning_rate: float,
                         decay: float,
                         logging_step: int,
                         lr_type: str,
                         warmup_ratio: float,
                         fp16: bool,
                         grad_accum: int,
                         progress_proxy,
                         interrupt_flag):
    """Запуск обучения в фоновом режиме."""
    progress_cb = _progress_callback_factory(progress_proxy)
    interrupt_check = lambda: bool(interrupt_flag.value)

    local_trainer = NERTrainer()
    try:
        results = local_trainer.train(
            dataset_name=dataset_name,
            annotation=annotation,
            model_name=model_name,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            decay=decay,
            logging_step=logging_step,
            lr_scheduler_type=lr_type,
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            gradient_accumulation_steps=grad_accum,
            progress_callback=progress_cb,
            interrupt_check=interrupt_check)

        if interrupt_flag.value:
            progress_proxy['status'] = 'interrupted'
        else:
            progress_proxy['status'] = 'completed'
            progress_proxy['metrics'] = results.get('metrics', {})
    except Exception as error:
        progress_proxy['status'] = 'error'
        progress_proxy['error'] = str(error)
    finally:
        interrupt_flag.value = False


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
    global training_process, training_progress, should_interrupt

    if training_process is not None and training_process.is_alive():
        return JSONResponse(status_code=400,
                            content={"message": "Обучение уже запущено"})

    contents = await dataset.read()
    annotation = json.loads(contents)

    # update_progress('not_started')
    training_progress['status'] = 'preparing'
    training_progress['progress'] = 0.0

    training_process = Process(target=run_training_process,
                               args=(dataset.filename,
                                     annotation,
                                     model_name,
                                     epochs,
                                     train_batch_size,
                                     eval_batch_size,
                                     learning_rate,
                                     decay,
                                     log,
                                     lr_type,
                                     warmup_radio,
                                     fp16,
                                     grad_accum,
                                     training_progress,
                                     should_interrupt))
    training_process.start()

    return {'message': 'Обучение началось'}

    # trainer.train(
    #     dataset_name=dataset.filename,
    #     annotation=annotation,
    #     model_name=model_name,
    #     epochs=epochs,
    #     train_batch_size=train_batch_size,
    #     eval_batch_size=eval_batch_size,
    #     learning_rate=learning_rate,
    #     decay=decay,
    #     logging_step=log,
    #     lr_scheduler_type=lr_type,
    #     warmup_ratio=warmup_radio,
    #     fp16=fp16,
    #     gradient_accumulation_steps=grad_accum
    # )
    # return {
    #     "message": "Обучение завершено успешно!",
    #     "metrics": results["metrics"]
    # }


@app.get('/training-progress')
async def get_training_progress():
    """Получение текущего прогресса обучения."""
    global training_progress
    progress_data = dict(training_progress)
    print('training_progress =', progress_data)
    return progress_data


@app.post('/interrupt-training')
async def interrupt_training():
    """Прерывание текущего обучения."""
    print("Был сделан запрос на прерывание обучения")
    global should_interrupt
    should_interrupt.value = True
    return {'message': 'Запрос на прерывание обучения отправлен'}
