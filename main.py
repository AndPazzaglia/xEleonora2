import streamlit as st
from tensorflow.keras.models import load_model
from utils.EvalModel import EvalModel
import os
import pickle
import bysp


# initial impagination
st.title('xEleonora')
st.write(
    'xEleonora is an AI model which generates italian poetries through a GPT2 fine tuning.')
st.sidebar.header('Instructions:')
st.sidebar.write(
    'Load model before using it.')

# choice of the author style
# auth_path = os.path.join('tokenization','clean_poetries_authors.pkl')
# with open(auth_path, 'rb') as f:
#     clean_poetries_authors = pickle.load(f)
# authors = clean_poetries_authors[0, :]
# authors = list(dict.fromkeys(authors))
# selected_author = st.sidebar.selectbox('Select the author style', authors, index=1)

# out_lines_number = st.sidebar.slider('Select the number of lines of the poetry', min_value=1, max_value=50, value=13, step=1)
# temperature = st.sidebar.slider('Select the model temperature (higher leads to more variable results)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# load model from splitted model
if st.sidebar.button('Load Model'):
    with st.spinner('Loading model...'):
        modelpath = os.path.join("models", "gpt2-poetries", "pytorch_model.bin")
        bysp.combine_file(modelpath)
    st.success('Model loaded')

st.sidebar.write(
    'Fill the following inputs and then hit the Run button!')
# text input
input_words = st.sidebar.text_input('Write initial words of the poetry', value="Il tuo sorriso è come")
if st.sidebar.button('Run model'):
    # load model
    model = "gpt2-poetries"
    # get output
    final_output = EvalModel(
        model, input_words=input_words)
    st.text(final_output)