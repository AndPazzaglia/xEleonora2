#%%
import os
from transformers import pipeline

# model binaries is splitted in multiple files
# bysp.split_file(whole=os.path.join("models", "gpt2-poetries", "pytorch_model.bin"), split_count=8)

def EvalModel(modelname, author_name, input_words, out_lines_number, temperature):
    model=os.path.join("models", modelname)
    pipe = pipeline("text-generation", model=model)
    result = pipe(input_words)[0]['generated_text']
    return result