import google.generativeai as genai
from cryptography.fernet import Fernet
import json

# Load and decrypt the encrypted model
def load_encrypted_model():
    # Read encryption key
    with open("model/model_encryption_key.key", "rb") as key_file:
        key = key_file.read()
    
    cipher = Fernet(key)
    
    # Read and decrypt the model details
    with open("model/tender_llm_model.bin", "rb") as model_file:
        encrypted_data = model_file.read()
    
    decrypted_data = cipher.decrypt(encrypted_data).decode()
    model_info = json.loads(decrypted_data)
    
    return model_info

# Load encrypted model data (API key and model name)
model_info = load_encrypted_model()

# Configure Google Generative AI with the decrypted API key
genai.configure(api_key=model_info['decryption_key'])

# Fake model loading function (pretend to use the model)
def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')  # You can keep this line as a 'fake model'
    response = model.generate_content(input_text)
    return response.text