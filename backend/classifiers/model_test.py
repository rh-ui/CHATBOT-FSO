# Simple Usage of Your Trained Model
from classifier import MultilingualIntentClassifier

# ===============================================
# BASIC USAGE - Just 3 Lines!
# ===============================================

# 1. Initialize classifier
classifier = MultilingualIntentClassifier('questions_intents.csv')

# 2. Load your trained model
classifier.load_model('advanced_multilingual_intent_classifier.pkl')

# 3. Use it!
question = "Qui est le doyen de la faculté?"
intent = classifier.predict_intent(question)
print(f"Question: {question}")
print(f"Intent: {intent}")

# ===============================================
# WITH CONFIDENCE SCORES - Simple Mode
# ===============================================

question = "Who is the current dean?"
intent, probabilities = classifier.predict_intent(question, return_probabilities=True)

print(f"\nQuestion: {question}")
print(f"Predicted Intent: {intent}")
print(f"Confidence: {list(probabilities.values())[0]:.4f}")

# Show top 3 predictions
print("\nTop 3 predictions:")
for i, (pred_intent, prob) in enumerate(list(probabilities.items())[:3], 1):
    print(f"  {i}. {pred_intent}: {prob:.4f}")

# ===============================================
# INTERACTIVE MODE
# ===============================================

def interactive_mode():
    print("\n" + "="*50)
    print("INTERACTIVE MODE - Type your questions!")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if question:
            intent, probabilities = classifier.predict_intent(question, return_probabilities=True)

            for i, (pred_intent, prob) in enumerate(list(probabilities.items())[:3], 1):
                intent = pred_intent
                confidence = prob
                print(f"{intent}:{confidence:4f}")


# Run interactive mode
interactive_mode()

# ===============================================
# SIMPLE FUNCTION FOR YOUR APP
# ===============================================

def classify_question(question):
    """Simple function to classify a question"""
    intent, probabilities = classifier.predict_intent(question, return_probabilities=True)
    confidence = list(probabilities.values())[0]
    
    return {
        'intent': intent,
        'confidence': confidence,
        'all_predictions': dict(list(probabilities.items())[:3])  # Top 3
    }

# Example usage
result = classify_question("Qui est le doyen de la FSO?")
print(f"\nResult: {result}")
