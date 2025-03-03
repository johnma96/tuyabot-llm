from tuyabot_llm import UseRAG
import gradio as gr

# Initialize the UseRAG object
rag = UseRAG()

# Create a Gradio Blocks interface
with gr.Blocks() as demo:
    # Add a Markdown title
    gr.Markdown("# Tuya BOT: Tu asistente para conocer nuestra companía")

    # Create a row with a column for the prompt
    with gr.Row():
        with gr.Column(scale=4):
            # Create a Textbox for the user's prompt
            prompt = gr.Textbox(label="Tu pregunta") 

            # Define some example prompts
            examples = [
                ["¿Cuáles son los valores numéricos de la tasa de interés y póliza del producto credicompras?"],
                ["¿Qué tipos de tarjeta de crédito ofrece TUYA?"] 
            ]

            # Add the examples to the interface
            gr.Examples(examples, inputs=prompt, label="Ejemplos de preguntas")            

    # Create an Accordion for advanced options
    with gr.Accordion("Opciones avanzadas", open=False): 
        # Create a Slider for the model's temperature
        temperature = gr.Slider(label="Temperatura del modelo", 
                                  minimum=0.1, 
                                  maximum=1, 
                                  value=0.2, 
                                  step=0.1, 
                                  info="Qué tan creativo quieres que sea?")

        # Create a row with two columns for additional options
        with gr.Row():
            with gr.Column():
                # Create a Slider for the top k value
                top_k = gr.Slider(label="Top k", 
                                     minimum=5, 
                                     maximum=50, 
                                     value=10, 
                                     step=1, 
                                     info="Cuántas opciones quiere que considere antes de responder?")

                # Create a Slider for the maximum length
                max_length = gr.Slider(label="Máxima longitud", 
                                          minimum=500, 
                                          maximum=2500, 
                                          value=2000, 
                                          step=100, 
                                          info="Cuál el máximo número de palabras que debo usar?")

            with gr.Column():
                # Create a Dropdown for the search type
                search_type = gr.Dropdown(choices=["similarity", "mmr", "similarity_score_threshold"], 
                                              value='similarity', 
                                              label="search_type", 
                                              info="Qué tipo de búsqueda uso para obtener contexto?")

                # Create a Slider for the size of the context
                size_context = gr.Slider(label="s_context", 
                                            minimum=5, 
                                            maximum=10, 
                                            value=5, 
                                            step=1, 
                                            info="Cuántos documentos quieres que use como contexto?")

    # Create a Textbox for the response
    output = gr.Textbox(label="Respuesta")

    # Create a row with two columns for the submit and clear buttons
    with gr.Row():
        with gr.Column(scale=4, min_width=50):
            # Create a Button to submit the prompt
            btn = gr.Button(value="Enviar", variant="primary") 

            # Define the function to call when the button is clicked
            btn.click(fn=rag.generate_response, 
                      inputs=[prompt, temperature, top_k, max_length, search_type, size_context], 
                      outputs=[output])

        with gr.Column(scale=4, min_width=50):
            # Create a Button to clear the prompt
            gr.ClearButton(components=[prompt], value="Limpiar prompt")

# Launch the Gradio interface
demo.launch(share=True)