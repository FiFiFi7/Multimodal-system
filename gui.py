import gradio as gr
from pipeline import AudioPipeline

pipeline = AudioPipeline()


def process_file(audio, mode, target_lang, target_emotion):
    return pipeline.run(audio, mode, target_lang, target_emotion)


gr.Interface(
    fn=process_file,
    inputs=[gr.Audio(type="filepath"), ...],
    outputs=gr.Audio(),
).launch()
