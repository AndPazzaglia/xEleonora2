#%%
import os
from tensorflow.keras.models import load_model
from utils.EvalModel import EvalModel

#%% load model
model_name = "gpt2-poetries"
model_path = os.path.join('models', model_name)
model = load_model(model_path)
final_output = EvalModel(
    model, input_words='il tuo sorriso è come')

print('--------------------------------------------------------')
print(final_output)
print('--------------------------------------------------------')
