from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from asgiref.wsgi import WsgiToAsgi
import nemo.collections.asr as nemo_asr
import torch

print("Loading ASR model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
model.to(device)
print("ASR model loaded.")
print("Device: {}".format(device))

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
asgi_app = WsgiToAsgi(app)

api_v2_cors_config = {
 "origins": ["*"],
 "methods": ["OPTIONS", "GET", "POST"],
 "allow_headers": ["Content-Type"]
}

@app.route('/', methods=['GET'])
def home():
    return 'Transcription Server is running!'

def get_transcribed_text(filepath):
    return model.transcribe(paths2audio_files=filepath)

@app.route('/api/file_to_transcribe', methods=['POST'])
@cross_origin(api_v2_cors_config)
def transcribe_files():
    try:
        outputs = {}
        if request.method == 'POST':
            print('POST request received')
            with open("input/original.wav","wb") as f:
                f.write(request.files['audioFile'].read())
            outputs['transcription'] = get_transcribed_text(["input/original.wav"])[0]
            outputs['message'] = 'Success'
            return jsonify({'result':outputs})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/directory_to_transcribe', methods=['POST'])
@cross_origin(api_v2_cors_config)
def transcribe_dirs():
    try:
        outputs = {}
        if request.method == 'POST':
            print('POST request received')
            files = request.json['filepaths']
            outputs['transcription'] = get_transcribed_text(files)[0]
            outputs['message'] = 'Success'
            return jsonify({'result':outputs})
    except Exception as e:
        return jsonify({"error": str(e)})

# if __name__=="__main__":
#     app.run(debug=True,host="0.0.0.0",port=8000)
