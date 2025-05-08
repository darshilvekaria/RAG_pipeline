import logging
import timeit

import box
import torch
import yaml
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('./config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

logging.info("setting dbqa")

if torch.cuda.is_available():
    logging.critical("CUDA is available")
else:
    logging.error("CUDA is not available")

# use this for for independent questions faster as there is no dependency of previous question
# def get_inference_response(input_q):
#     # Setup DBQA
#     print(f'Your Query: {input_q}')
#     start = timeit.default_timer()

#     print('Setting up DB QA')
#     dbqa = setup_dbqa()
#     print('[IN PROGRESS] Send Query')
#     response = dbqa.invoke({'query': input_q})
#     print('[COMPLETED] Query Processing')
#     print(f'\nAnswer: {response["result"]}')

#     # If you're expecting `source_documents`, you'll need to also return them from your chain
#     # Otherwise, below lines will fail.
#     # For now, this will only work if `dbqa` is set up to return a dict with `result` and `source_documents`
#     source_docs = response['source_documents']
#     output_res = response["result"]

#     for i, doc in enumerate(source_docs):
#         # print(f'\nSource Document {i+1}\n')
#         output_res += '\n____________________________________________________________________'
#         output_res += f'\nSource Document {i + 1}\n'
#         output_res += f'Source Text: {doc.page_content}\n'
#         output_res += f'Document Name: {doc.metadata["source"]}\n'
#         output_res += f'Page Number: {doc.metadata["page"]}'
#         # print(f'Source Text: {doc.page_content}')
#         # print(f'Document Name: {doc.metadata["source"]}')
#         # print(f'Page Number: {doc.metadata["page"]}\n')
#     end = timeit.default_timer()
#     # global response_time 
#     response_time = end-start
#     print(f"Time to retrieve response: {end - start}")


#     return output_res, response_time


# Maintain chat history, use this for dependent questions from the previous chat
chat_history = []

def get_inference_response(input_q):
    global chat_history

    print(f'Your Query: {input_q}')
    start = timeit.default_timer()

    print('Setting up Conversational Retrieval Chain...')
    dbqa = setup_dbqa()

    print('[IN PROGRESS] Sending Query...')
    response = dbqa.invoke({
        'question': input_q,
        'chat_history': chat_history
    })
    print('[COMPLETED] Query Processed')

    # Update chat history
    chat_history.append((input_q, response["answer"]))

    output_res = response["answer"]
    source_docs = response.get("source_documents", [])

    for i, doc in enumerate(source_docs):
        output_res += '\n' + ('_' * 70)
        output_res += f'\nSource Document {i + 1}\n'
        output_res += f'Source Text: {doc.page_content}\n'
        output_res += f'Document Name: {doc.metadata.get("source")}\n'
        output_res += f'Page Number: {doc.metadata.get("page")}'

    end = timeit.default_timer()
    response_time = end - start
    print(f"Time to retrieve response: {response_time:.2f} seconds")

    return output_res, response_time
