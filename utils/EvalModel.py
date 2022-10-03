#%%
import os
from transformers import pipeline

def EvalModel(model, author_name, input_words, out_lines_number, temperature):
    pipe = pipeline("text-generation", model=os.path.join("models", model))
    result = pipe(input_words)[0]['generated_text']
    return result