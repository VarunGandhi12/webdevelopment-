from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Initialize Flask app
app = Flask(__name__)

# Define the chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    # Get the user's message from the request
    user_input = request.json.get("message")
    
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # Generate a response
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the response as JSON
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)