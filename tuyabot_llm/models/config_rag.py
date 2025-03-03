import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from langchain_huggingface import HuggingFacePipeline

from tuyabot_llm import AbsolutePaths
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

class ConfigRAG:
    """
    Class for configuring and using a Retrieve-and-Generate (RAG) language model.

    Attributes
    ----------
    llm : HuggingFacePipeline
        Language model used for text generation.
    retriever : Chroma
        Retrieval model used for obtaining relevant documents.
    default_llm_config : dict
        Default configuration for the language model.
    default_retriever_config : dict
        Default configuration for the retrieval model.
    """

    def __init__(self):
        """
        Initializes the ConfigRAG class.
        """
        pass

    def config_model(self, temperature: float = 0.2, top_k: int = 10, max_length: int = 2000) -> HuggingFacePipeline:
        """
        Configures the language model.

        Parameters
        ----------
        temperature : float, optional
            Temperature for text generation (default is 0.2).
        top_k : int, optional
            Number of top-k tokens to consider for text generation (default is 10).
        max_length : int, optional
            Maximum length for text generation (default is 2000).

        Returns
        -------
        HuggingFacePipeline
            Configured language model.
        """
    

        model_id = 'unsloth/Llama-3.2-1B-Instruct' 
        tokenizer = AutoTokenizer.from_pretrained(model_id, device="cuda:0", truncation=True)
        model = AutoModelForCausalLM.from_pretrained(model_id) 


        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            truncation=True
            # repetition_penalty=1.5,
            # no_repeat_ngram_size=4,  # Ajusta el tamaño de los n-gramas que no se pueden repetir
        )

        #conversión a uso api tipo langchain
         
        return HuggingFacePipeline(pipeline=pipe)
    
    def config_retriever(self, search_type: str = "similarity", size_context: int = 5) -> Chroma:
        """
        Configures the retrieval model.

        Parameters
        ----------
        search_type : str, optional
            Type of search to perform (default is "similarity").
        size_context : int, optional
            Size of the context to consider for search (default is 5).

        Returns
        -------
        Chroma
            Configured retrieval model.
        """

        search_kwargs={"k": size_context}

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        current_dir = AbsolutePaths().get_abs_path_folder('raw')
        db_dir = os.path.join(current_dir, "tuya_collection")
        persistent_directory_clean = os.path.join(db_dir, "chroma_tuya_collection_cleaned")

        db = Chroma(persist_directory=persistent_directory_clean, embedding_function=embeddings)
        
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        return retriever

    def use_retriever(self, query: str, **retriever_kwargs: dict) -> list:
        """
        Uses the retrieval model to obtain relevant documents.

        Parameters
        ----------
        query : str
            Query to search for.
        **retriever_kwargs
            Additional keyword arguments for the retrieval model.

        Returns
        -------
        list
            Relevant documents obtained.
        """

        retriever = self.config_retriever(**retriever_kwargs)

        relevant_docs = retriever.invoke(query)

        return relevant_docs
    
    def format_docs(self, docs: list) -> str:
        """
        Formats the obtained documents.

        Parameters
        ----------
        docs : list
            Documents to format.

        Returns
        -------
        str
            Formatted documents.
        """

        formatted_docs = "\n".join(doc.page_content for doc in docs)
        # print(f"**Formatted Docs**: {formatted_docs}\n*******************************")  # Inspeccionar la salida de format_docs
        return formatted_docs
    
    def check_current_rag(self, llm_kwargs: dict, retriever_kwargs: dict) -> None:
        """
        Checks if the language model or retrieval model needs to be re-instantiated.

        Parameters
        ----------
        llm_kwargs : dict
            Keyword arguments for the language model.
        retriever_kwargs : dict
            Keyword arguments for the retrieval model.
        """

        if "llm" in self.__dict__:
            # print(llm_kwargs, self.default_llm_config)
            if llm_kwargs != self.default_llm_config:
                print('Reinstanciando el llm')
                self.llm = self.config_model(**llm_kwargs)
                self.default_llm_config = llm_kwargs

        else:
            print('Instanciando llm por primera vez')
            self.llm = self.config_model(**self.default_llm_config)

        if 'retriever' in self.__dict__:
            # print(retriever_kwargs, self.default_retriever_config)
            if retriever_kwargs != self.default_retriever_config:
                print('Reinstanciando el retriever')
                self.retriever = self.config_retriever(**retriever_kwargs)
                self.default_retriever_config = retriever_kwargs

        else:
            print('Instanciando retriever por primera vez')
            self.retriever = self.config_retriever(**self.default_retriever_config)