'''
===========================================
        Module: Util functions
===========================================
'''
import logging

import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm
from src.InstructorEmbeddingWrapper import InstructorEmbeddingWrapper
from functools import lru_cache

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))



@lru_cache(maxsize=None)
def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

# use build_conversational_chain for dependent questions from the previous chat
def build_conversational_chain(llm, retriever, prompt_template):
    # QA Chain
    question_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    qa_chain = LLMChain(llm=llm, prompt=question_prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context"
    )

    # Question Generator (to reformulate follow-ups into standalone Qs)
    condense_question_prompt = PromptTemplate.from_template(
        """Given the conversation and a follow-up question, rephrase the follow-up to be a standalone question.

        Chat History:
        {chat_history}
        Follow-up Input: {question}
        Standalone question:"""
    )
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt)

    # Conversational Chain with both pieces
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True
    )

    return chain

################################################################
# use build_retrieval_qa for independent questions faster as there is no dependency of previous question

# def build_retrieval_qa(llm, prompt, vectordb):
#     dbqa = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
#                                        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
#                                        chain_type_kwargs={'prompt': prompt}
#                                        )
#     return dbqa


################################################################
# # use it for sentence transformer model while using all-MiniLM-L6-v2 for db_build

# @lru_cache(maxsize=None)
# def setup_dbqa():  
#     embeddings = HuggingFaceEmbeddings(model_name="models/all-MiniLM-L6-v2",
#                                     model_kwargs={'device': 'cuda'}
#                                     )
#     vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     qa_prompt = set_qa_prompt()
#     llm = build_llm()
#     dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
#     logging.info("Hugging Face Embedding Loaded")

#     return dbqa

################################################################
# use setup_dbqa for independent questions faster as there is no dependency of previous question

# def setup_dbqa():
#     embeddings = InstructorEmbeddingWrapper(model_path="models_instructor/instructor-base", device="cuda")
    
#     vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
#     qa_prompt = set_qa_prompt()
#     llm = build_llm()
#     dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

#     logging.info("Instructor Embeddings Loaded")

#     return dbqa


################################################################
# use setup_dbqa for dependent questions from the previous chat
def setup_dbqa():
    embeddings = InstructorEmbeddingWrapper(model_path="models_instructor/instructor-base", device="cuda")
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})
    llm = build_llm()
    prompt = set_qa_prompt()

    return build_conversational_chain(llm, retriever, prompt.template)