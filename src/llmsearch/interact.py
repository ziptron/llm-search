# /usr/local/lib/python3.10/dist-packages/llmsearch - Location on Colab
import langchain
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from termcolor import cprint

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import Config, ResponseModel
from llmsearch.process import get_and_parse_response

load_dotenv()
langchain.debug = True


def print_llm_response(output: ResponseModel):
    print("\n============= SOURCES ==================")
    for source in output.semantic_search:
        source.metadata.pop("source")
        cprint(source.chunk_link, "blue")
        cprint(source.metadata, "cyan")
        print("******************* BEING EXTRACT *****************")
        print(f"{source.chunk_text}\n")
    print("\n============= RESPONSE =================")
    cprint(output.response, "red")
    print("------------------------------------------")


def qa_with_llm(llm, prompt: str, config: Config, chain_type="stuff", max_k=12):
    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path), embeddings_model_config=config.embeddings.embedding_model)
    embed_retriever = store.load_retriever(
        search_type=config.semantic_search.search_type, search_kwargs={"k": max_k}
    )

    chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt=prompt)

    while True:
        
        question = input("\nENTER COMMAND >> ")
        if question.lower() == "extract":
            with open('prompt.txt', 'r') as file:
                question = file.read()
        output = get_and_parse_response(
            query=question,
            chain=chain,
            embed_retriever=embed_retriever,
            config=config.semantic_search,
        )
        print_llm_response(output)
