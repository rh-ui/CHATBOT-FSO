# Complete Usage Example for Advanced Multilingual Intent Classifier

# First, make sure you have all required packages installed:
# pip install pandas scikit-learn numpy matplotlib seaborn textblob joblib

from classifier import MultilingualIntentClassifier
import pandas as pd

def main_example():
    classifier = MultilingualIntentClassifier('questions_intents.csv')  # Your 2MB CSV file
    
    # Step 2: Load and preprocess data
    data = classifier.load_and_preprocess_data()
    
    classifier.prepare_features()
    
    classifier.split_data(test_size=0.2, random_state=42)
    best_combo, best_score = classifier.train_models()
    ensemble, ensemble_accuracy = classifier.create_ensemble_model()
    classifier.evaluate_best_model()
    classifier.test_multilingual_examples()
    classifier.save_model('advanced_multilingual_intent_classifier.pkl')
    return classifier

def usage_examples(classifier):
    question = "Qui est actuellement le doyen de la faculté des Sciences?"
    intent = classifier.predict_intent(question)
    print(f"Question: {question}")
    print(f"Predicted Intent: {intent}")
    
    # Example 2: Prediction with probabilities
    print("\n2. Prediction with Confidence Scores:")
    question = "Who is the current dean?"
    intent, probabilities = classifier.predict_intent(question, return_probabilities=True)
    print(f"Question: {question}")
    print(f"Predicted Intent: {intent}")
    print("Top 5 predictions with confidence:")
    for i, (pred_intent, prob) in enumerate(list(probabilities.items())[:5], 1):
        print(f"  {i}. {pred_intent}: {prob:.4f}")
    
    # Example 3: Batch predictions
    print("\n3. Batch Predictions:")
    test_questions = [
        "Qui est le doyen de la faculté?",  # French
        "Who is the dean of the faculty?",  # English  
        "من هو عميد الكلية؟",  # Arabic
        "Anwa i d amenzu n temdint?",  # Amazigh
        "Comment contacter le doyen?",  # French - contact
        "What are the dean's office hours?",  # English - hours
    ]
    
    for i, question in enumerate(test_questions, 1):
        intent, probs = classifier.predict_intent(question, return_probabilities=True)
        top_intent, top_prob = list(probs.items())[0]
        print(f"  {i}. '{question}' → {intent} (confidence: {top_prob:.3f})")

def interactive_session():
    # Load the trained model
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    try:
        classifier.load_model('advanced_multilingual_intent_classifier.pkl')
    except:
        return
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if question:
                # Get prediction with probabilities
                intent, probabilities = classifier.predict_intent(question, return_probabilities=True)
                
                print(f"\nPredicted Intent: {intent}")
                
                # Show top 5 predictions with visual bars
                top_predictions = list(probabilities.items())[:5]
                for i, (pred_intent, prob) in enumerate(top_predictions, 1):
                    # Create visual confidence bar
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    print(f"  {i}. {pred_intent}")
                    print(f"     {bar} {prob:.4f}")
                    
        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def load_and_use_pretrained():
    # Create classifier instance
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    
    try:
        classifier.load_model('advanced_multilingual_intent_classifier.pkl')
        test_cases = [
            "Qui est le doyen?",
            "Dean contact information",
            "معلومات الاتصال بالعميد",
            "Office hours of the dean"
        ]
        
        for question in test_cases:
            intent, probs = classifier.predict_intent(question, return_probabilities=True)
            confidence = list(probs.values())[0]
            print(f"'{question}' → {intent} ({confidence:.3f})")
            
    except FileNotFoundError:
        print("No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def model_performance_analysis(classifier):
    sorted_models = sorted(classifier.models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, info) in enumerate(sorted_models[:10], 1):  # Top 10
        print(f"{i:2d}. {name:30} : {info['accuracy']:.4f}")
    
    # Show best vectorizer and classifier combination
    vectorizer_performance = {}
    classifier_performance = {}
    
    for name, info in classifier.models.items():
        vec_name = name.split('_')[0] + '_' + name.split('_')[1]  # e.g., tfidf_bigram
        clf_name = '_'.join(name.split('_')[2:])  # e.g., naive_bayes
        
        if vec_name not in vectorizer_performance:
            vectorizer_performance[vec_name] = []
        if clf_name not in classifier_performance:
            classifier_performance[clf_name] = []
            
        vectorizer_performance[vec_name].append(info['accuracy'])
        classifier_performance[clf_name].append(info['accuracy'])
    
    print("\nAverage Performance by Vectorizer:")
    for vec, scores in vectorizer_performance.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {vec:20} : {avg_score:.4f}")
    
    print("\nAverage Performance by Classifier:")
    for clf, scores in classifier_performance.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {clf:20} : {avg_score:.4f}")

# Complete workflow example
if __name__ == "__main__":
    
    classifier = main_example()
    
    # Usage examples
    usage_examples(classifier)
    
    # Performance analysis
    model_performance_analysis(classifier)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Interactive testing session")
        print("2. Load and test pre-trained model")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_session()
        elif choice == '2':
            load_and_use_pretrained()
        elif choice == '3':
            print("Thank you for using ibtissam Classifier")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def quick_predict(question, model_path='advanced_multilingual_intent_classifier.pkl'):
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    classifier.load_model(model_path)
    return classifier.predict_intent(question, return_probabilities=True)

# Example of integration into your application
def integrate_into_app():
    # Your application receives a user question
    user_question = "Qui est le doyen de la FSO?"
    
    # Get prediction
    intent, probabilities = quick_predict(user_question)
    
    # Use the prediction in your application logic
    if intent == "fso doyen nom mot-du-doyen":
        response = "Je vais vous donner les informations sur le doyen de la FSO..."
    else:
        response = f"Question classifiée comme: {intent}"
    
    return response
