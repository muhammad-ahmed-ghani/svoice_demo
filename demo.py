from svoice.separate import *
import scipy.io as sio
from scipy.io.wavfile import write
import gradio as gr
import os
from transformers import AutoProcessor, pipeline
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from glob import glob

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.makedirs('input', exist_ok=True)
os.makedirs('separated', exist_ok=True)
os.makedirs('whisper_checkpoint', exist_ok=True)

print("Loading ASR model...")
processor = AutoProcessor.from_pretrained("openai/whisper-small")
if not os.path.exists("whisper_checkpoint"):
    model = ORTModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small", from_transformers=True)
    speech_recognition_pipeline = pipeline(
    "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )
    model.save_pretrained("whisper_checkpoint")
else:
    model = ORTModelForSpeechSeq2Seq.from_pretrained("whisper_checkpoint", from_transformers=False)
    speech_recognition_pipeline = pipeline(
    "automatic-speech-recognition",
        model=model,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )
print("Whisper ASR model loaded.")

def separator(audio, rec_audio):
    outputs= {}

    if audio:
        write('input/original.wav', audio[0], audio[1])
    elif rec_audio:
        write('input/original.wav', rec_audio[0], rec_audio[1])

    separate(mix_dir="./input")
    separated_files = glob(os.path.join('separated', "*.wav"))
    separated_files = [f for f in separated_files if "original.wav" not in f]
    outputs['transcripts'] = []
    for file in sorted(separated_files):
        separated_audio = sio.wavfile.read(file)
        outputs['transcripts'].append(speech_recognition_pipeline(separated_audio[1])['text'])
    return sorted(separated_files) + outputs['transcripts']
    
def set_example_audio(example: list) -> dict:
    return gr.Audio.update(value=example[0])

demo = gr.Blocks()
with demo:
    gr.Markdown('''
    <center>
        <h1>Multiple Voice Separation with Transcription DEMO</h1>
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
    button.click(separator, inputs=[input_audio, rec_audio], outputs=outputs_audio + outputs_text)

demo.launch()