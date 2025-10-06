"""FastAPI - приложение для NER."""

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ner_model import NERModel


TEXT = "text"

app = FastAPI()
model = NERModel()


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
    return FileResponse('model_page.html')


