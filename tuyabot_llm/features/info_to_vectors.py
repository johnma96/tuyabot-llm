import os

from langchain.embeddings import SentenceTransformerEmbeddings
from tuyabot_llm import AbsolutePaths, GetInfoWebPages
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


class InfoToVectors(GetInfoWebPages):
    """
    A class used to process and vectorize information.

    Attributes
    ----------
    model_name : str
        The name of the model to use for embeddings.
    main_collection : str
        The name of the main collection.
    collection_name : str
        The name of the collection.
    embeddings : SentenceTransformerEmbeddings
        The embeddings object used for vectorization.
    """

    def __init__(self,
                model_name_to_embeddings: str = "all-MiniLM-L6-v2", 
                main_collection: str = "tuya_collection",
                collection_name: str = "chroma_tuya_collection_cleaned"):
        """
        Initialize the InfoToVectors class.

        Parameters
        ----------
        model_name_to_embeddings : str, optional
            The name of the model to use for embeddings (default is "all-MiniLM-L6-v2").
        main_collection : str, optional
            The name of the main collection (default is "tuya_collection").
        collection_name : str, optional
            The name of the collection (default is "chroma_tuya_collection_cleaned").
        """
        
        self.model_name = model_name_to_embeddings
        self.main_collection = main_collection
        self.collection_name = collection_name

        self.embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)

    def load_original_data(self, query: str = 'Tuya', 
                           search_type: str = 'similarity',
                           num_documents: int = 10) -> list:
        """
        Load the original data from the Chroma vector store.

        Parameters
        ----------
        query : str, optional
            The query to search for (default is 'Tuya').
        search_type : str, optional
            The type of search to perform (default is 'similarity').
        num_documents : int, optional
            The number of documents to retrieve (default is 10).

        Returns
        -------
        list
            A list of documents containing the retrieved information.
        """

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
    def chunk_files(documents: list, 
                    chunk_size: int = 600,
                    chunk_overlap: int = 40,
                    length_function = len) -> list:
        """
        Chunk the given documents into smaller pieces.

        Parameters
        ----------
        documents : list
            A list of documents to chunk.
        chunk_size : int, optional
            The size of each chunk (default is 600).
        chunk_overlap : int, optional
            The overlap between chunks (default is 40).
        length_function : callable, optional
            The function to use to calculate the length of each chunk (default is len).

        Returns
        -------
        list
            A list of chunked documents.
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )

        return text_splitter.split_documents(documents)
        

    def clean_info(self, documents: list = None) -> list:
        """
        Clean the given documents by lowercasing the text, removing special characters, and chunking the text.

        Parameters
        ----------
        documents : list, optional
            A list of documents to clean (default is None, which loads the original data).

        Returns
        -------
        list
            A list of cleaned and chunked documents.
        """

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
    
    def process_original_info(self) -> Chroma:
        """
        Process the original information by cleaning and vectorizing it, and save it to a new Chroma vector store.

        Returns
        -------
        Chroma
            The new Chroma vector store containing the processed information.
        """

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