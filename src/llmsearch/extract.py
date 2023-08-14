# /usr/local/lib/python3.10/dist-packages/llmsearch - Location on Colab
import langchain
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

    #while True:
    #question = input("\nENTER QUESTION >> ")

    question = """
    The these are insurance claim notes. From these claim notes, output the following information in JSON format using the categories as follows.
        "Claim amount" is the total amount of damages in dollars.
        "Insured name" is the name of the insured, a person or entity.
        "Claim type" is classified into one of these categories: Water Damage or Fire Damage or Vehicle Impact or Impact by another object.
        "Component that caused the damage" is the physical component, item, product, or system that caused the damages.
        "Make and model of Component that caused the damage" – if applicable, is the make and model of the component; otherwise, say unknown
        "Cause of Loss" explains why this failure occurred
        "Was the failed component retained as evidence" indicates whether the failed component was retained and is a Yes/No/Unknown
        "Location of Evidence" if the component was retained, where is it and with whom?
        "Was this a trailer attachment" describes whether the component was an attachment to a trailer and is a Yes/No/Unknown
        "Was it original or an after market part" describe if this component was 'Original' to the trailer, was an 'Aftermarket' part, or write 'Unknown'
        "At-fault third party" - If there is a very likely responsible party that caused the loss, name the party; if no one, say unknown; if a manufacturer or installer is the at-fault party, but the name is unknown, then just write manufacturer or installer depending on which was at fault."
    """






    output = get_and_parse_response(
        query=question,
        chain=chain,
        embed_retriever=embed_retriever,
        config=config.semantic_search,
    )
    print_llm_response(output)
