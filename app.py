from tuyabot_llm import UseRAG
import gradio as gr

rag = UseRAG()

with gr.Blocks() as demo:
    gr.Markdown("# Tuya BOT: Tu asistente para conocer nuestra companía")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Tu pregunta") #Give prompt some real estate
            examples = [
                ["¿Cuáles son los valores numéricos de la tasa de interés y póliza del producto credicompras?"],
                ["¿Qué tipos de tarjeta de crédito ofrece TUYA?"] 
            ]
            gr.Examples(examples, inputs=prompt, label="Ejemplos de preguntas")            
    with gr.Accordion("Opciones avanzadas", open=False): #Let's hide the advanced options!
            temperature = gr.Slider(label="Temperatura del modelo",minimum=0.1,maximum=1,value=0.2,step=0.1,info="Qué tan creativo quieres que sea?")
            with gr.Row():
                with gr.Column():
                    top_k=gr.Slider( label="Top k", minimum=5, maximum=50, value=10, step=1, info="Cuántas opciones quiere que considere antes de responder?")
                    max_length=gr.Slider( label="Máxima longitud", minimum=500, maximum=2500, value=2000, step=100, info="Cuál el máximo número de palabras que debo usar?")
                with gr.Column():
                    search_type=gr.Dropdown(choices=["similarity", "mmr", "similarity_score_threshold"], value='similarity', label="search_type", info="Qué tipo de búsqueda uso para obtener contexto?")
                    size_context=gr.Slider(label="s_context", minimum=5, maximum=10, value=5, step=1, info="Cuántos documentos quieres que use como contexto?")
    output = gr.Textbox(label="Respuesta")
    with gr.Row():
        with gr.Column(scale=4, min_width=50):
            btn = gr.Button(value="Enviar", variant="primary") #Submit button side by side!         
            btn.click(fn=rag.generate_response, inputs=[prompt,temperature,top_k,max_length,search_type,size_context], outputs=[output])
        with gr.Column(scale=4, min_width=50):
            gr.ClearButton(components=[prompt], value="Limpiar prompt")

demo.launch(share=True)