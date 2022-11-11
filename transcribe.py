from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from asgiref.wsgi import WsgiToAsgi
import whisper
from glob import glob

print("Loading ASR model...")
model = whisper.load_model("base")
device = "cuda"
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
            path = request.json['filepaths']
            files = glob(path+"/*")
            outputs['transcription'] = []
            for i in files:
                outputs['transcription'].extend(model.transcribe("audio.mp3")["text"])
            outputs['message'] = 'Success'
            return jsonify({'result':outputs})
    except Exception as e:
        return jsonify({"error": str(e)})

# if __name__=="__main__":
#     app.run(debug=True,host="0.0.0.0",port=8000)
