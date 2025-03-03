import gradio as gr
from src.app.model_handler import LanguageDetector
from src.app.example_data import EXAMPLE_SENTENCES
from src.app.config import GRADIO_CONFIG

def create_gradio_interface():
    detector = LanguageDetector()
    
    demo = gr.Interface(
        fn=detector.predict_language,
        inputs=[
            gr.Textbox(label="Enter text", placeholder="Type a sentence here..."),
            gr.Slider(1, 10, value=5, step=1, label="Top-k Languages"),
        ],
        outputs=gr.Textbox(label="Predicted Languages"),
        title=GRADIO_CONFIG["title"],
        description=GRADIO_CONFIG["description"],
        examples=[[sent, 5] for sent in EXAMPLE_SENTENCES],
        flagging_mode="manual"
    )
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()