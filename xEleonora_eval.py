#%%
import os
from utils.EvalModel import EvalModel

#%% load model
model_name = "gpt2-poetries"
final_output = EvalModel(
    model_name, input_words='il tuo sorriso è come')

print('--------------------------------------------------------')
print(final_output)
print('--------------------------------------------------------')
