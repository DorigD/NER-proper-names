from main import NER
from utils.config import DATA_DIR
from datasets import load_from_disk
import os

ner = NER()
data = os.path.join(DATA_DIR, "raw")
for file in os.listdir(data):
    model=ner.train(training_data=os.path.join(data,file))
    ner.load_model(model)