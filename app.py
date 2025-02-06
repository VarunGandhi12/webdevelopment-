from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__, static_folder='../frontend/build')
CORS(app)

# Load model
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-gpt2")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    inputs = tokenizer.encode(data['message'] + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response.split(tokenizer.eos_token)[-1].strip()})

# Serve frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)