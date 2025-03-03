import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from langchain_huggingface import HuggingFacePipeline

from tuyabot_llm import AbsolutePaths
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

class ConfigRAG:
    def __init__(self):
        pass

    def config_model(self, temperature: float = 0.2, top_k: int = 10, max_length: int = 2000):
    
        model_id = 'unsloth/Llama-3.2-1B-Instruct' 
        tokenizer = AutoTokenizer.from_pretrained(model_id, device="cuda:0", truncation=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device) 

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            truncation=True,
            device=device
            # repetition_penalty=1.5,
            # no_repeat_ngram_size=4,  # Ajusta el tamaño de los n-gramas que no se pueden repetir
        )

        #conversión a uso api tipo langchain
         
        return HuggingFacePipeline(pipeline=pipe)
    
    def config_retriever(self, search_type="similarity", size_context=5):

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

    def use_retriever(self, query, **retriever_kwargs):

        retriever = self.config_retriever(**retriever_kwargs)

        relevant_docs = retriever.invoke(query)

        return relevant_docs
    
    def format_docs(self, docs):
        formatted_docs = "\n".join(doc.page_content for doc in docs)
        # print(f"**Formatted Docs**: {formatted_docs}\n*******************************")  # Inspeccionar la salida de format_docs
        return formatted_docs
    
    def check_current_rag(self, llm_kwargs, retriever_kwargs):

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