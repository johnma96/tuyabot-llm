import os

from langchain.embeddings import SentenceTransformerEmbeddings
from tuyabot_llm import AbsolutePaths, GetInfoWebPages
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


class InfoToVectors(GetInfoWebPages):
    def __init__(self,
                model_name_to_embeddings: str = "all-MiniLM-L6-v2", 
                main_collection: str = "tuya_collection",
                collection_name: str = "chroma_tuya_collection_cleaned"):
        
        self.model_name = model_name_to_embeddings
        self.main_collection = main_collection
        self.collection_name = collection_name

        self.embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)

    def load_original_data(self, query: str = 'Tuya', 
                           search_type: str = 'similarity',
                           num_documents: int = 10):

        current_dir = AbsolutePaths().get_abs_path_folder('raw')
        db_dir = os.path.join(current_dir, self.main_collection)
        persistent_directory_original = os.path.join(db_dir, self.collection_name)

        db = Chroma(persist_directory=persistent_directory_original, 
                    embedding_function=self.embeddings)
        
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": num_documents},
        )

        return retriever.invoke(query)
    
    @staticmethod
    def chunk_files(documents, 
                    chunk_size=600,
                    chunk_overlap=40,
                    length_function=len):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )

        return text_splitter.split_documents(documents)
        

    def clean_info(self, documents = None):

        if documents is None:
            documents = self.load_original_data()
        
        # Lower text
        for doc in documents:
            doc.page_content = doc.page_content.lower()

        # Clean some special characters
        for document in documents:
            document.page_content = document.page_content.strip()
            document.page_content = document.page_content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            document.page_content = ' '.join(document.page_content.split())

        return InfoToVectors.chunk_files(documents=documents)
    
    def process_original_info(self):

        original_docs = self.load_original_data()
        cleaned_docs = self.clean_info(documents=original_docs)

        # Save in new vd
        db = GetInfoWebPages.pages_info_to_vd(
            documents=cleaned_docs,
            model_name=self.model_name,
            main_collection=self.main_collection,
            collection_name=self.collection_name
        )

        return db