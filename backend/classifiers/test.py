# Complete Usage Example for Advanced Multilingual Intent Classifier

# First, make sure you have all required packages installed:
# pip install pandas scikit-learn numpy matplotlib seaborn textblob joblib

from classifier import MultilingualIntentClassifier
import pandas as pd

def main_example():
    """
    Complete example of using the Advanced Multilingual Intent Classifier
    """
    
    print("🚀 Advanced Multilingual Intent Classifier Example")
    print("=" * 60)
    
    # Step 1: Initialize the classifier with your CSV file
    print("Step 1: Initializing classifier...")
    classifier = MultilingualIntentClassifier('questions_intents.csv')  # Your 2MB CSV file
    
    # Step 2: Load and preprocess data
    print("\nStep 2: Loading and preprocessing data...")
    data = classifier.load_and_preprocess_data()
    
    # Display data info
    print(f"✅ Loaded {len(data)} questions")
    print(f"✅ Found {data['intent'].nunique()} unique intents")
    print("\nSample data:")
    print(data.head())
    
    # Step 3: Prepare different feature extraction methods
    print("\nStep 3: Preparing feature extractors...")
    classifier.prepare_features()
    print("✅ Created 4 different vectorizers (TF-IDF unigram, bigram, trigram, Count)")
    
    # Step 4: Split data into training and testing
    print("\nStep 4: Splitting data...")
    classifier.split_data(test_size=0.2, random_state=42)
    print("✅ Data split into 80% training, 20% testing")
    
    # Step 5: Train multiple models
    print("\nStep 5: Training multiple models...")
    print("This will test 16 combinations (4 vectorizers × 4 classifiers)")
    best_combo, best_score = classifier.train_models()
    print(f"✅ Best combination: {best_combo}")
    print(f"✅ Best accuracy: {best_score:.4f}")
    
    # Step 6: Create ensemble model (optional but recommended)
    print("\nStep 6: Creating ensemble model...")
    ensemble, ensemble_accuracy = classifier.create_ensemble_model()
    print(f"✅ Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Step 7: Detailed evaluation
    print("\nStep 7: Detailed evaluation...")
    classifier.evaluate_best_model()
    
    # Step 8: Test with multilingual examples
    print("\nStep 8: Testing with multilingual examples...")
    classifier.test_multilingual_examples()
    
    # Step 9: Save the trained model
    print("\nStep 9: Saving model...")
    classifier.save_model('advanced_multilingual_intent_classifier.pkl')
    print("✅ Model saved successfully!")
    
    return classifier

def usage_examples(classifier):
    """
    Show various ways to use the trained classifier
    """
    
    print("\n" + "="*60)
    print("🎯 USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Single prediction
    print("\n1. Single Prediction:")
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
    """
    Interactive session for testing the classifier
    """
    
    print("\n" + "="*60)
    print("💬 INTERACTIVE SESSION")
    print("="*60)
    print("Type your questions in any language (French, English, Arabic, Amazigh)")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    # Load the trained model
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    try:
        classifier.load_model('advanced_multilingual_intent_classifier.pkl')
        print("✅ Model loaded successfully!")
    except:
        print("❌ No pre-trained model found. Please train the model first.")
        return
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question:
                # Get prediction with probabilities
                intent, probabilities = classifier.predict_intent(question, return_probabilities=True)
                
                print(f"\n🎯 Predicted Intent: {intent}")
                print("📊 Confidence Distribution:")
                
                # Show top 5 predictions with visual bars
                top_predictions = list(probabilities.items())[:5]
                for i, (pred_intent, prob) in enumerate(top_predictions, 1):
                    # Create visual confidence bar
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    print(f"  {i}. {pred_intent}")
                    print(f"     {bar} {prob:.4f}")
                    
                # Show if confidence is low
                if top_predictions[0][1] < 0.5:
                    print("⚠️  Low confidence - you might want to rephrase your question")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def load_and_use_pretrained():
    """
    Example of loading and using a pre-trained model
    """
    
    print("\n" + "="*60)
    print("📦 LOADING PRE-TRAINED MODEL")
    print("="*60)
    
    # Create classifier instance
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    
    # Load pre-trained model
    try:
        classifier.load_model('advanced_multilingual_intent_classifier.pkl')
        print("✅ Pre-trained model loaded successfully!")
        
        # Test some predictions
        test_cases = [
            "Qui est le doyen?",
            "Dean contact information",
            "معلومات الاتصال بالعميد",
            "Office hours of the dean"
        ]
        
        print("\nTesting loaded model:")
        for question in test_cases:
            intent, probs = classifier.predict_intent(question, return_probabilities=True)
            confidence = list(probs.values())[0]
            print(f"'{question}' → {intent} ({confidence:.3f})")
            
    except FileNotFoundError:
        print("❌ No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

def model_performance_analysis(classifier):
    """
    Analyze the performance of different models
    """
    
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Show all model performances
    print("All Model Combinations Performance:")
    sorted_models = sorted(classifier.models.items(), 
                          key=lambda x: x[1]['accuracy'], 
                          reverse=True)
    
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
    
    print("🎉 Welcome to Advanced Multilingual Intent Classifier!")
    print("This example will show you all the features step by step.")
    
    # Full training and evaluation
    print("\n" + "🔥" * 60)
    print("TRAINING PHASE")
    print("🔥" * 60)
    
    classifier = main_example()
    
    # Usage examples
    usage_examples(classifier)
    
    # Performance analysis
    model_performance_analysis(classifier)
    
    # Ask user what they want to do next
    print("\n" + "🎮" * 60)
    print("INTERACTIVE OPTIONS")
    print("🎮" * 60)
    
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
            print("👋 Thank you for using the Advanced Multilingual Intent Classifier!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

# Quick prediction function for direct use
def quick_predict(question, model_path='advanced_multilingual_intent_classifier.pkl'):
    """
    Quick prediction function - use this for integrating into your applications
    """
    classifier = MultilingualIntentClassifier('questions_intents.csv')
    classifier.load_model(model_path)
    return classifier.predict_intent(question, return_probabilities=True)

# Example of integration into your application
def integrate_into_app():
    """
    Example of how to integrate the classifier into your application
    """
    
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

print("\n" + "ℹ️" * 60)
print("HOW TO RUN THIS EXAMPLE:")
print("ℹ️" * 60)
print("1. Save this code as 'example_usage.py'")
print("2. Make sure you have the main classifier code saved as 'multilingual_intent_classifier.py'")
print("3. Put your CSV file 'questions_intents.csv' in the same directory")
print("4. Run: python example_usage.py")
print("5. Follow the interactive prompts!")
print("\nFor quick predictions in your app, use the quick_predict() function!")