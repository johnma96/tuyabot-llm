import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from tuyabot_llm import AbsolutePaths

from langchain.embeddings import SentenceTransformerEmbeddings

class GetInfoWebPages:
    """
    A class used to retrieve and process information from web pages.

    Attributes
    ----------
    urls : list
        A list of URLs to retrieve information from.
    model_name : str, optional
        The name of the model to use for embeddings (default is "all-MiniLM-L6-v2").
    main_collection : str, optional
        The name of the main collection (default is "tuya_collection").
    collection_name : str, optional
        The name of the collection (default is "chroma_tuya_collection").
    db : Chroma
        The Chroma vector store containing the processed information.
    """

    def __init__(self, urls: str | list, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 main_collection: str = "tuya_collection",
                 collection_name: str = "chroma_tuya_collection") -> None:
        """
        Initialize the GetInfoWebPages class.

        Parameters
        ----------
        urls : str | list
            A URL or list of URLs to retrieve information from.
        model_name : str, optional
            The name of the model to use for embeddings (default is "all-MiniLM-L6-v2").
        main_collection : str, optional
            The name of the main collection (default is "tuya_collection").
        collection_name : str, optional
            The name of the collection (default is "chroma_tuya_collection").
        """

        if isinstance(urls, str): [urls]

        self.urls = urls
        self.model_name = model_name
        self.main_collection = main_collection
        self.collection_name = collection_name

        docs = GetInfoWebPages.get_information(urls=self.urls)
        self.db = GetInfoWebPages.pages_info_to_vd(
            documents=docs,
            model_name=self.model_name, 
            main_collection=self.main_collection, 
            collection_name=self.collection_name
        )

    @staticmethod
    def get_information(urls: list) -> list:
        """
        Retrieve information from the given URLs.

        Parameters
        ----------
        urls : list
            A list of URLs to retrieve information from.

        Returns
        -------
        list
            A list of documents containing the retrieved information.
        """

        # Create a loader for web content
        loader = WebBaseLoader(urls)
        documents = loader.load()

        return documents

    @staticmethod
    def pages_info_to_vd(documents: list, 
                         model_name: str = "all-MiniLM-L6-v2", 
                         main_collection: str = "tuya_collection",
                         collection_name: str = "chroma_tuya_collection") -> Chroma:
        """
        Process the retrieved information and store it in a Chroma vector store.

        Parameters
        ----------
        documents : list
            A list of documents containing the retrieved information.
        model_name : str, optional
            The name of the model to use for embeddings (default is "all-MiniLM-L6-v2").
        main_collection : str, optional
            The name of the main collection (default is "tuya_collection").
        collection_name : str, optional
            The name of the collection (default is "chroma_tuya_collection").

        Returns
        -------
        Chroma
            The Chroma vector store containing the processed information.
        """
        
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)

        current_dir = AbsolutePaths().get_abs_path_folder('raw')
        db_dir = os.path.join(current_dir, main_collection)
        persistent_directory = os.path.join(db_dir, collection_name)

        if not os.path.exists(persistent_directory):
            print(f"\n--- Creating vector store in {persistent_directory} ---")
            db = Chroma.from_documents(documents, embeddings, persist_directory=persistent_directory)
            print(f"--- Finished creating vector store in {persistent_directory} ---")
        else:
            print(f"Vector store {persistent_directory} already exists. No need to initialize.")
            db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        return db