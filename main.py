from svoice.separate import *
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from asgiref.wsgi import WsgiToAsgi
import os
import requests
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
asgi_app = WsgiToAsgi(app)

os.makedirs('input', exist_ok=True)
os.makedirs('separated', exist_ok=True)

api_v2_cors_config = {
 "origins": ["*"],
 "methods": ["OPTIONS", "GET", "POST"],
 "allow_headers": ["Content-Type"]
}

@app.route('/', methods=['GET'])
def home():
    return 'Speech Separation Server is running!'

@app.route('/api/file_to_separate',methods = ['POST', 'GET'])
@cross_origin(**api_v2_cors_config)
def seperator():
    try:
        outputs = {}
        if request.method=='POST':
            with open("input/original.wav","wb") as f:
                f.write(request.files['audioFile'].read())
            outputs['path'] = separate(mix_dir="./input")
            res = requests.post('http://localhost:8000/api/directory_to_transcribe', json={'filepaths': outputs['path']})
            outputs['transcription'] = json.loads(res.text)['result']['transcription']
            outputs['message'] = "Success"
            return jsonify({'result':outputs})

    except Exception as e:
        return jsonify({"error": str(e)})

# if __name__=="__main__":
#     app.run(debug=True,host="0.0.0.0",port=5000)
