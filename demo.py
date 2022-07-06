from svoice.separate import *
from scipy.io.wavfile import write
import gradio as gr
import os
import requests
import json
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.makedirs('input', exist_ok=True)
os.makedirs('separated', exist_ok=True)

def separator(audio, rec_audio):
    outputs= {}

    if audio:
        write('input/original.wav', audio[0], audio[1])
    elif rec_audio:
        write('input/original.wav', rec_audio[0], rec_audio[1])

    outputs['path'] = separate(mix_dir="./input")
    res = requests.post('http://localhost:8000/api/directory_to_transcribe', json={'filepaths': outputs['path']})
    outputs['transcription'] = json.loads(res.text)['result']['transcription']
    return outputs['path'] + outputs['transcription']
    
def set_example_audio(example: list) -> dict:
    return gr.Audio.update(value=example[0])

demo = gr.Blocks()
with demo:
    gr.Markdown('''
    <center>
        <h1>Multiple Voice Separation with Transcription DEMO</h1>
        <h2>Final Year Project Group ID: F21BS003</h2>
        <div style="display:flex;align-items:center;justify-content:center;"><iframe src="https://streamable.com/e/0x8osl?autoplay=1&nocontrols=1" frameborder="0" allow="autoplay"></iframe></div>
        <p>
            This is a demo for the multiple voice separation algorithm. The algorithm is trained on the LibriMix7 dataset and can be used to separate multiple voices from a single audio file.
        </p>
    </center>
    ''')
    
    with gr.Row():
        input_audio = gr.Audio(label="Input audio", type="numpy")
        rec_audio = gr.Audio(label="Record Using Microphone", type="numpy", source="microphone")

    with gr.Row():
        output_audio1 = gr.Audio(label='Speaker 1', interactive=False)
        output_text1 = gr.Text(label='Speaker 1', interactive=False)
        output_audio2 = gr.Audio(label='Speaker 2', interactive=False)
        output_text2 = gr.Text(label='Speaker 2', interactive=False)

    with gr.Row():
        output_audio3 = gr.Audio(label='Speaker 3', interactive=False)
        output_text3 = gr.Text(label='Speaker 3', interactive=False)
        output_audio4 = gr.Audio(label='Speaker 4', interactive=False)
        output_text4 = gr.Text(label='Speaker 4', interactive=False)

    with gr.Row():
        output_audio5 = gr.Audio(label='Speaker 5', interactive=False)
        output_text5 = gr.Text(label='Speaker 5', interactive=False)
        output_audio6 = gr.Audio(label='Speaker 6', interactive=False)
        output_text6 = gr.Text(label='Speaker 6', interactive=False)

    with gr.Row():
        output_audio7 = gr.Audio(label='Speaker 7', interactive=False)
        output_text7 = gr.Text(label='Speaker 7', interactive=False)

    outputs_audio = [output_audio1, output_audio2, output_audio3, output_audio4, output_audio5, output_audio6, output_audio7]
    outputs_text = [output_text1, output_text2, output_text3, output_text4, output_text5, output_text6, output_text7]
    button = gr.Button("Separate")
    # with gr.Row():
    #     example_audios = gr.Dataset(components=[input_audio],
    #                                 samples=[[BASE_PATH+'/samples/test1.wav'], [BASE_PATH+'/samples/test2.wav'], [BASE_PATH+'/samples/test3.wav'], [BASE_PATH+'/samples/test4.wav'], [BASE_PATH+'/samples/test5.wav']])
    # example_audios.click(fn=set_example_audio, inputs=example_audios, outputs=example_audios.components)
    button.click(separator, inputs=[input_audio, rec_audio], outputs=outputs_audio + outputs_text)

demo.launch(server_port=5000, share=True)
