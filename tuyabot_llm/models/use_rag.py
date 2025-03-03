from .config_rag import ConfigRAG

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class UseRAG(ConfigRAG):
    def __init__(self, t = 0.2, t_k = 10, m_length = 2000, s_type = 'similarity', s_context = 5):

        self.t = t
        self.t_k = t_k
        self.m_length = m_length
        self.s_type = s_type
        self.s_context = s_context

        self.default_llm_config = {'temperature': self.t, 'top_k': self.t_k, 'max_length': self.m_length}
        self.default_retriever_config = {'search_type': self.s_type, 'size_context': self.s_context}

        super().__init__()

        self.check_current_rag(llm_kwargs=self.default_llm_config, retriever_kwargs=self.default_retriever_config)

    def generate_response(self, question, temperature=None, top_k=None, max_length=None, search_type=None, size_context=None):

        if temperature is None: temperature = self.t
        if top_k is None: top_k = self.t_k
        if max_length is None: max_length = self.m_length
        if search_type is None: search_type = self.s_type
        if size_context is None: size_context = self.s_context

        llm_kwargs = {'temperature': temperature, 'top_k': top_k, 'max_length': max_length}
        retriever_kwargs = {'search_type': search_type, 'size_context': size_context}

        self.check_current_rag(llm_kwargs=llm_kwargs, retriever_kwargs=retriever_kwargs)

        self.template = """
            Eres un agente de servicio al cliente que trabaja para TUYA SA, una empresa que se
            dedica a ser la solución financiera del retail y que busca apoyar a los sectores
            vulnerables de la sociedad. 

            Como agente de servicio al cliente, debes suministrar respuestas amigables y
            claras a los clientes.

            Emplea el contexto que te ofrece la empresa TUYA delimitado por triple comillas invertidas, para responder
            la pregunta que se encuentra al final delimitada por comillas simples.

            Siempre que puedas responder con una serie de items hazlo, tu respuesta es máximo de 15 palabras y debe ser completamente 
            en español.

            Contexto: ```{context}```

            Pregunta: '{question}'
            """
        
        prompt = PromptTemplate.from_template(self.template)

        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        output = rag_chain.invoke(question)

        output = 'Pregunta: '+ '\n'.join([element.strip() for element in output.split("Pregunta:")[-1].split("'\n")])

        return  output