# ## ..................Mistral prompt 
qa_template = """[INST] Use the following pieces of information to answer the user's question. 

If you do not have enough relevant information in the context to provide an accurate answer, just say that you don't know, don't try to make up an answer. 
Provide a descriptive answer unless the question is very specific or you are asked to be brief. Only return the helpful answer below and nothing else.
Don't add any follow up questions on your own.
[INST]
Context: {context}
Question: {question}
[/INST]
Answer:
"""