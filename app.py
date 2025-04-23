from flask import Flask, request, jsonify
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_PATH = "Model"
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Global variables for model and tokenizer
model = None
tokenizer = None

# Model loading
def load_model():
    global model, tokenizer
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        logger.info("Loading configuration...")
        config = PeftConfig.from_pretrained(MODEL_PATH)
        
        logger.info(f"Loading base model: {config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model when the application starts
@app.before_request
def initialize():
    global model, tokenizer
    if model is None:
        success = load_model()
        if not success:
            logger.error("Failed to load model at startup")

# API endpoint for generating responses
@app.route('/generate', methods=['POST'])
def generate_response():
    global model, tokenizer
    
    # Get request data
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    prompt = data['prompt']
    logger.info(f"Received prompt: {prompt[:50]}...")
    
    # Optional parameters with defaults
    max_length = data.get('max_length', DEFAULT_MAX_LENGTH)
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    top_p = data.get('top_p', DEFAULT_TOP_P)
    
    try:
        # Ensure model is loaded
        if model is None or tokenizer is None:
            success = load_model()
            if not success:
                return jsonify({'error': 'Failed to load model'}), 500
                
        # Generate text
        logger.info("Generating response...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        logger.info(f"Generated response: {response[:50]}...")
        
        return jsonify({
            'prompt': prompt,
            'response': response
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating response: {error_msg}")
        return jsonify({'error': error_msg}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    global model
    status = "healthy" if model is not None else "model not loaded"
    return jsonify({'status': status})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)