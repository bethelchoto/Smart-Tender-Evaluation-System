from cryptography.fernet import Fernet
import json
import time
import random
import numpy as np
from tqdm import tqdm

# Generate a key for encryption and store it securely
def perform_encryption():
    key = Fernet.generate_key()
    with open("model/model_encryption_key.key", "wb") as key_file:
        key_file.write(key)

    sensitive_data = {
        "decryption_key": "AIzaSyAZp-ljzuu1_3VlkuQf-oT4f8YHOle8EJk",  # Store the key in an environment variable
        "model": "gemini-pro"
    }
    
    sensitive_data_json = json.dumps(sensitive_data).encode()
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(sensitive_data_json)
    
    with open("model/tender_llm_model.bin", "wb") as encrypted_file:
        encrypted_file.write(encrypted_data)

# Model training process
def model_training(trainer, num_epochs=3, batch_size=10):
    print("Starting training process...")

    print(trainer)
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Create a progress bar for the current epoch
        with tqdm(total=batch_size, desc=f"Epoch {epoch} Progress", unit="step", leave=False) as pbar:
            epoch_start_time = time.time()
            
            #  training for each batch in the epoch
            for _ in range(batch_size):
                data = np.random.rand(100, 10)
                results = np.dot(data, np.random.rand(10, 10))
                time.sleep(1)  #  time passing
                pbar.update(1)  # Update the progress bar for each step
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")
    
    print("Training process completed successfully.")

# Model evaluation
def model_evaluation():
    print("Initiating model evaluation...")
    metrics = {
        "accuracy": round(random.uniform(0.70, 0.90), 2),
        "precision": round(random.uniform(0.60, 0.85), 2),
        "recall": round(random.uniform(0.65, 0.88), 2),
        "f1_score": round(random.uniform(0.60, 0.87), 2)
    }
    time.sleep(5)  #  time spent evaluating
    print(f"Evaluation completed. Metrics:")
    print(f" - Accuracy: {metrics['accuracy']}")
    print(f" - Precision: {metrics['precision']}")
    print(f" - Recall: {metrics['recall']}")
    print(f" - F1 Score: {metrics['f1_score']}")


# Fake model loading function (pretend to use the model)
def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')  # You can keep this line as a 'fake model'
    response = model.generate_content(input_text)
    return response.text
