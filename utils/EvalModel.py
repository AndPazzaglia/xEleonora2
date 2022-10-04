#%%
import os
from transformers import pipeline, AutoTokenizer

def EvalModel(modelname, input_words, author_name=None, out_lines_number=None, temperature=None):
    model = os.path.join("models", modelname)
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = pipe(input_words)[0]['generated_text'].replace(" .", ".")
    return result