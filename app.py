import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title('Natural Language Generation with GPT-2')
st.markdown("A [simple demonstration](https://github.com/CaliberAI/streamlit-get-stories-aylien) of using [Streamlit](https://streamlit.io/) with [HuggingFace's GPT-2](https://github.com/huggingface/transformers/).")

seed = st.text_input('Seed', 'The dog jumped')
num_return_sequences = st.number_input('Number of generated sequences', 1, 100, 20)
max_length = st.number_input('Length of sequences', 5, 100, 20)
go = st.button('Generate')

if go:
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode(seed)).unsqueeze(0)
        output = model.generate(input_ids=input_ids,
                                max_length=max_length,
                                num_return_sequences=num_return_sequences,
                                do_sample=True,
                                length_penalty=10)
        sequences = []
        for j in range(len(output)):
          for i in range(len(output[j])):
              sequences.append(tokenizer.decode(output[j][i].tolist(), skip_special_tokens=True))
        st.dataframe(sequences)
    except Exception as e:
        st.exception("Exception: %s\n" % e)

st.markdown('___')
st.markdown('by [CaliberAI](https://github.com/CaliberAI/)')
