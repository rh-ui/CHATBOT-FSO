import fasttext

# Load the language identification model
model = fasttext.load_model("C:/Users/Maarouf/Desktop/.venv/CHATBOT-FSO/backend/lang.bin")

# Example text
text = "smi"

# Predict language
predictions = model.predict(text)

# Extract ISO code and confidence
language_code = predictions[0][0]
confidence = predictions[1][0]

print(f"Detected language: {language_code}")
print(f"Confidence: {confidence:.2f}")
