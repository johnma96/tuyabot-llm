
Informe de Desarrollo de Producto de IA Generativa para TUYA SA
===========================================================
# Resumen
Este informe describe el desarrollo de un producto de IA generativa que proporciona información contextualizada sobre los productos financieros de TUYA SA. El producto utiliza una arquitectura RAG (Retrieve, Augment, Generate) que combina un modelo LLM (Large Language Model) con una base de datos vectorial para generar respuestas personalizadas. A lo largo del texto encontrará links útiles que se han consultado para el desarrollo de la prueba.

# Esquema de desarrollo
**1. Creación del Repositorio en GitHub:**
Se creó un repositorio en GitHub para almacenar el código del proyecto. El repositorio se estructuró siguiendo las mejores prácticas de desarrollo de código, incluyendo la creación de ramas (branches) para diferentes etapas del desarrollo.

**2. Estructuración del Proyecto:**
Se utilizó la una [propuesta](https://github.com/johnma96/machine_learning_project_template) basada en la dada por [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) para estructurar el proyecto de manera lógica y fácil de navegar. Esto incluyó la creación de carpetas para datos, notebooks, modelos y aplicaciones.

**3. Desarrollo del Producto de IA generativa:**
El desarrollo del producto se dividió en varias etapas, cada una de las cuales se realizó en un notebook separado para facilitar la reproducción del desarrollo.


- **3.1. Obtención de Información de Páginas Web**
Se obtuvo la información de las páginas web de TUYA SA utilizando técnicas de scraping web con [BeatifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) y con [langchain](https://python.langchain.com/docs/how_to/document_loader_web/).
- **3.2. Configuración del Modelo LLM**
Se descargó, instaló y probó el modelo LLM [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) en un pipeline de Langchain y Hugging Face. Se configuró el modelo para ejecutarse en GPU.
- **3.3. Procesamiento de Información y Creación de Base de Datos Vectorial**
Se procesó la información (texto) extraída de las páginas web utilizando [embeddings de Langchain](https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/). Luego, se creó una base de datos vectorial en local utilizando [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/).
- **3.4. Definición del Template Prompt y Proceso RAG**
Se definió un template prompt y se configuró un proceso [RAG](https://python.langchain.com/docs/tutorials/rag/) que toma una pregunta, busca en la base de datos vectorial, trae los documentos relacionados, los pasa como contexto al prompt y luego genera una respuesta utilizando el modelo LLM.
- **3.5. Desarrollo de la Aplicación en Gradio**
Se desarrolló una aplicación en [Gradio](https://www.gradio.app/) utilizando un template sencillo y luego se personalizó utilizando el contexto Blocks.
3.6. Documentación y Integración de Ramas en GIT
Se documentaron todas las funcionalidades y se integraron las ramas en GIT.

# Próximos Pasos
1. Colocar la aplicación de Gradio en la nube a través de los [Spaces de Hugging Face](https://huggingface.co/docs/hub/spaces-overview) y una base de datos vectorial en [Milvus](https://milvus.io/) o alguna solución de código abierto.
2. Evaluar cómo se integran los [LLMs con MLflow](https://mlflow.org/docs/latest/llms/index.html) para hacer tracking de modelos.
3. Desarrollar un esquema de [CI/CD](https://latitude.so/blog/best-practices-for-llm-observability-in-cicd/) con GitHub Actions o alguna solución que lo permita.
