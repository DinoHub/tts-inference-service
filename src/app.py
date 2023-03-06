import logging

import gradio as gr
from config import BaseConfig
from predict import inputs, outputs, predict

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
    config = BaseConfig()

    app = gr.Interface(
        predict,
        inputs=inputs,
        outputs=outputs,
        title="TTS Inference Service",
        description="TTS Inference Service for AI App Store",
    )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        enable_queue=True
    )
    
