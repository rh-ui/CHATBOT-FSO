import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import pickle
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings
warnings.filterwarnings('ignore')

class MultilingualIntentClassifier:
    def __init__(self, csv_file_path):
        """
        Initialize the classifier with data from CSV file
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.vectorizers = {}
        self.best_model = None
        self.best_vectorizer = None
        
    def load_and_preprocess_data(self):
        """
        Load data from CSV with robust parsing to handle commas in questions
        """
        print("Loading data from CSV...")
        
        try:
            # First, try standard pandas read_csv
            self.data = pd.read_csv(self.csv_file_path, encoding='utf-8')
            print("✅ Standard CSV parsing successful")
            
        except pd.errors.ParserError as e:
            print(f"⚠️ Standard parsing failed: {str(e)}")
            print("🔧 Trying robust CSV parsing...")
            
            # Robust parsing method
            questions = []
            intents = []
            
            try:
                with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Find the last comma to split question from intent
                    # This assumes the intent doesn't contain commas
                    if ',' in line:
                        last_comma_idx = line.rfind(',')
                        question = line[:last_comma_idx].strip()
                        intent = line[last_comma_idx + 1:].strip()
                        
                        # Remove quotes if present
                        question = question.strip('"\'')
                        intent = intent.strip('"\'')
                        
                        if question and intent:
                            questions.append(question)
                            intents.append(intent)
                    else:
                        print(f"⚠️ Skipping malformed line {line_num}: {line[:50]}...")
                
                # Create DataFrame
                self.data = pd.DataFrame({
                    'question': questions,
                    'intent': intents
                })
                
                print("✅ Robust CSV parsing successful")
                
            except Exception as e:
                print(f"❌ Robust parsing also failed: {str(e)}")
                print("🔧 Trying alternative method...")
                
                # Alternative method using csv module
                questions = []
                intents = []
                
                with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                    # Try different delimiters and quoting
                    for delimiter in [',', ';', '\t']:
                        file.seek(0)
                        try:
                            csv_reader = csv.reader(file, delimiter=delimiter, quotechar='"')
                            temp_questions = []
                            temp_intents = []
                            
                            for row_num, row in enumerate(csv_reader, 1):
                                if len(row) >= 2:
                                    # If more than 2 columns, join all but last as question
                                    if len(row) > 2:
                                        question = ','.join(row[:-1]).strip()
                                        intent = row[-1].strip()
                                    else:
                                        question = row[0].strip()
                                        intent = row[1].strip()
                                    
                                    if question and intent:
                                        temp_questions.append(question)
                                        temp_intents.append(intent)
                                elif len(row) == 1 and ',' in row[0]:
                                    # Single field with comma - split manually
                                    parts = row[0].rsplit(',', 1)
                                    if len(parts) == 2:
                                        question = parts[0].strip()
                                        intent = parts[1].strip()
                                        if question and intent:
                                            temp_questions.append(question)
                                            temp_intents.append(intent)
                            
                            if len(temp_questions) > 100:  # If we got reasonable data
                                questions = temp_questions
                                intents = temp_intents
                                print(f"✅ Alternative parsing successful with delimiter '{delimiter}'")
                                break
                                
                        except Exception:
                            continue
                
                if not questions:
                    raise ValueError("❌ All parsing methods failed. Please check your CSV format.")
                
                self.data = pd.DataFrame({
                    'question': questions,
                    'intent': intents
                })
        
        # Ensure we have the right column names
        if self.data.shape[1] == 2:
            self.data.columns = ['question', 'intent']
        
        # Remove any rows with missing data
        initial_count = len(self.data)
        self.data = self.data.dropna()
        final_count = len(self.data)
        
        if initial_count != final_count:
            print(f"🧹 Removed {initial_count - final_count} rows with missing data")
        
        print(f"📊 Loaded {len(self.data)} samples")
        print(f"📊 Number of unique intents: {self.data['intent'].nunique()}")
        
        # Display intent distribution
        intent_counts = self.data['intent'].value_counts()
        print(f"📊 Most common intent: '{intent_counts.index[0]}' ({intent_counts.iloc[0]} samples)")
        print(f"📊 Least common intent: '{intent_counts.index[-1]}' ({intent_counts.iloc[-1]} samples)")
        
        # Show sample data
        print("\n📋 Sample data:")
        for i in range(min(3, len(self.data))):
            question = self.data.iloc[i]['question']
            intent = self.data.iloc[i]['intent']
            print(f"  Q: {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"  I: {intent}")
            print()
        
        # Clean the text data
        print("🧹 Cleaning text data...")
        self.data['question_clean'] = self.data['question'].apply(self.clean_text)
        
        return self.data
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Keep accented characters for multilingual support
        # Remove only specific punctuation that might not be useful
        text = re.sub(r'[^\w\s\u00C0-\u017F\u0600-\u06FF\u2D30-\u2D7F?]', ' ', text)
        
        return text
    
    def prepare_features(self):
        """
        Prepare different feature extraction methods
        """
        print("🔧 Preparing feature extractors...")
        
        # TF-IDF with different n-gram ranges
        self.vectorizers = {
            'tfidf_unigram': TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.95,
                strip_accents=None,  # Keep accents for multilingual
                lowercase=True
            ),
            'tfidf_bigram': TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents=None,
                lowercase=True
            ),
            'tfidf_trigram': TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                strip_accents=None,
                lowercase=True
            ),
            'count_vectorizer': CountVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents=None,
                lowercase=True
            )
        }
        
        print("✅ Created 4 different vectorizers")
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets with intelligent handling of small classes
        """
        print("📊 Splitting data...")
        
        X = self.data['question_clean']
        y = self.data['intent']
        
        # Analyze class distribution
        intent_counts = y.value_counts()
        min_samples = intent_counts.min()
        single_sample_classes = intent_counts[intent_counts == 1]
        
        print(f"📊 Intent distribution analysis:")
        print(f"   Total intents: {len(intent_counts)}")
        print(f"   Min samples per intent: {min_samples}")
        print(f"   Max samples per intent: {intent_counts.max()}")
        print(f"   Intents with only 1 sample: {len(single_sample_classes)}")
        
        if len(single_sample_classes) > 0:
            print(f"⚠️ Found {len(single_sample_classes)} intents with only 1 sample")
            
            # Strategy 1: Move single-sample intents to training set
            single_sample_mask = y.isin(single_sample_classes.index)
            multi_sample_mask = ~single_sample_mask
            
            # Split only the multi-sample data
            if multi_sample_mask.sum() > 0:
                X_multi = X[multi_sample_mask]
                y_multi = y[multi_sample_mask]
                
                # Check if we can stratify the multi-sample data
                multi_intent_counts = y_multi.value_counts()
                if multi_intent_counts.min() >= 2:
                    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
                        X_multi, y_multi, test_size=test_size, random_state=random_state, stratify=y_multi
                    )
                    print("✅ Used stratified split for multi-sample intents")
                else:
                    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
                        X_multi, y_multi, test_size=test_size, random_state=random_state
                    )
                    print("✅ Used random split for multi-sample intents")
                
                # Add all single-sample intents to training set
                X_single = X[single_sample_mask]
                y_single = y[single_sample_mask]
                
                # Combine training sets
                self.X_train = pd.concat([X_train_multi, X_single])
                self.X_test = X_test_multi
                self.y_train = pd.concat([y_train_multi, y_single])
                self.y_test = y_test_multi
                
                print("✅ Added single-sample intents to training set")
                
            else:
                # All intents have only 1 sample - use simple random split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=min(test_size, 0.1), random_state=random_state  # Reduce test size
                )
                print("⚠️ All intents have single samples, used minimal test split")
        
        elif min_samples >= 2:
            # Standard stratified split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print("✅ Used stratified split")
        else:
            # Fallback to random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print("✅ Used random split")
        
        print(f"📊 Final split results:")
        print(f"   Training set size: {len(self.X_train)}")
        print(f"   Test set size: {len(self.X_test)}")
        print(f"   Training intents: {self.y_train.nunique()}")
        print(f"   Test intents: {self.y_test.nunique()}")
        
        # Verify no empty sets
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            raise ValueError("❌ Data split resulted in empty training or test set!")
            
        return True
        
    def train_models(self):
        """
        Train multiple models with different vectorizers
        """
        print("🚀 Training models...")
        print("This will test 16 combinations (4 vectorizers × 4 classifiers)")
        
        # Define classifiers
        classifiers = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'svm': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
        }
        
        best_score = 0
        best_combo = None
        
        # Train each combination of vectorizer and classifier
        for vec_name, vectorizer in self.vectorizers.items():
            print(f"\n🔍 Testing with {vec_name}...")
            
            try:
                # Fit vectorizer on training data
                X_train_vec = vectorizer.fit_transform(self.X_train)
                X_test_vec = vectorizer.transform(self.X_test)
                
                for clf_name, classifier in classifiers.items():
                    print(f"  🤖 Training {clf_name}...")
                    
                    try:
                        # Train classifier
                        classifier.fit(X_train_vec, self.y_train)
                        
                        # Predict and evaluate
                        y_pred = classifier.predict(X_test_vec)
                        accuracy = accuracy_score(self.y_test, y_pred)
                        
                        combo_name = f"{vec_name}_{clf_name}"
                        self.models[combo_name] = {
                            'vectorizer': vectorizer,
                            'classifier': classifier,
                            'accuracy': accuracy
                        }
                        
                        print(f"    ✅ Accuracy: {accuracy:.4f}")
                        
                        # Track best model
                        if accuracy > best_score:
                            best_score = accuracy
                            best_combo = combo_name
                            self.best_model = classifier
                            self.best_vectorizer = vectorizer
                            
                    except Exception as e:
                        print(f"    ❌ Error training {clf_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"  ❌ Error with {vec_name}: {str(e)}")
                continue
        
        if best_combo:
            print(f"\n🏆 Best model: {best_combo} with accuracy: {best_score:.4f}")
        else:
            raise ValueError("❌ No models were trained successfully!")
        
        return best_combo, best_score
    
    def create_ensemble_model(self):
        """
        Create an ensemble of the best performing models
        """
        print("🎪 Creating ensemble model...")
        
        if len(self.models) < 2:
            print("⚠️ Not enough models for ensemble, skipping...")
            return None, 0
        
        # Select top 3 models
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_models = sorted_models[:min(3, len(sorted_models))]
        
        print("🥇 Top models for ensemble:")
        for name, model_info in top_models:
            print(f"  📊 {name}: {model_info['accuracy']:.4f}")
        
        try:
            # Create ensemble
            estimators = []
            for i, (name, model_info) in enumerate(top_models):
                estimators.append((f'model_{i}', model_info['classifier']))
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Use probability-based voting
                n_jobs=1  # Reduced from -1 to avoid issues
            )
            
            # Train ensemble with best vectorizer
            X_train_vec = self.best_vectorizer.transform(self.X_train)
            ensemble.fit(X_train_vec, self.y_train)
            
            # Evaluate ensemble
            X_test_vec = self.best_vectorizer.transform(self.X_test)
            ensemble_pred = ensemble.predict(X_test_vec)
            ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
            
            print(f"🎪 Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            # If ensemble is better, use it as best model
            current_best_accuracy = max(model['accuracy'] for model in self.models.values())
            if ensemble_accuracy > current_best_accuracy:
                self.best_model = ensemble
                print("🏆 Using ensemble as best model")
            
            return ensemble, ensemble_accuracy
            
        except Exception as e:
            print(f"❌ Error creating ensemble: {str(e)}")
            return None, 0
    
    def evaluate_best_model(self):
        """
        Detailed evaluation of the best model
        """
        print("\n📈 Detailed evaluation of best model:")
        
        try:
            X_test_vec = self.best_vectorizer.transform(self.X_test)
            y_pred = self.best_model.predict(X_test_vec)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"🎯 Accuracy: {accuracy:.4f}")
            
            # Classification report for manageable number of classes
            unique_intents = self.data['intent'].nunique()
            if unique_intents <= 50:  # Only show detailed report for reasonable number of classes
                print("\n📊 Classification Report:")
                print(classification_report(self.y_test, y_pred, zero_division=0))
            else:
                print(f"📊 Too many classes ({unique_intents}) for detailed report")
                
                # Show performance for top intents only
                top_intents = self.data['intent'].value_counts().head(10).index
                test_mask = self.y_test.isin(top_intents)
                
                if test_mask.sum() > 0:
                    print("\n📊 Classification Report (Top 10 intents):")
                    y_test_top = self.y_test[test_mask]
                    y_pred_top = pd.Series(y_pred)[test_mask]
                    print(classification_report(y_test_top, y_pred_top, zero_division=0))
            
        except Exception as e:
            print(f"❌ Error in evaluation: {str(e)}")
    
    def predict_intent(self, question, return_probabilities=False):
        """
        Predict intent for a new question
        """
        if self.best_model is None or self.best_vectorizer is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Clean the question
        clean_question = self.clean_text(question)
        
        # Vectorize
        question_vec = self.best_vectorizer.transform([clean_question])
        
        # Predict
        predicted_intent = self.best_model.predict(question_vec)[0]
        
        if return_probabilities:
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(question_vec)[0]
                intent_probs = dict(zip(self.best_model.classes_, probabilities))
                # Sort by probability
                intent_probs = dict(sorted(intent_probs.items(), key=lambda x: x[1], reverse=True))
                return predicted_intent, intent_probs
            else:
                return predicted_intent, {predicted_intent: 1.0}
        
        return predicted_intent
    
    def save_model(self, model_path='intent_classifier_model.pkl'):
        """
        Save the trained model and vectorizer
        """
        try:
            model_data = {
                'vectorizer': self.best_vectorizer,
                'classifier': self.best_model,
                'intent_classes': list(self.data['intent'].unique()) if self.data is not None else []
            }
            
            joblib.dump(model_data, model_path)
            print(f"💾 Model saved to {model_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self, model_path='intent_classifier_model.pkl'):
        """
        Load a pre-trained model
        """
        try:
            model_data = joblib.load(model_path)
            self.best_vectorizer = model_data['vectorizer']
            self.best_model = model_data['classifier']
            print(f"📂 Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def test_multilingual_examples(self):
        """
        Test the model with multilingual examples
        """
        print("\n🌍 Testing with multilingual examples:")
        
        test_questions = [
            "Qui est le doyen de la faculté?",  # French
            "Who is the dean of the faculty?",  # English
            "من هو عميد الكلية؟",  # Arabic
            "Anwa i d amenzu n temdint?",  # Amazigh
            "What is the name of the current dean?",  # English variation
            "Comment contacter le doyen?",  # French variation
        ]
        
        for question in test_questions:
            try:
                intent, probabilities = self.predict_intent(question, return_probabilities=True)
                top_3_intents = list(probabilities.items())[:3]
                
                print(f"\n❓ Question: {question}")
                print(f"🎯 Predicted Intent: {intent}")
                print("📊 Top 3 probabilities:")
                for i, (intent_name, prob) in enumerate(top_3_intents, 1):
                    print(f"  {i}. {intent_name}: {prob:.4f}")
                    
            except Exception as e:
                print(f"❌ Error predicting for '{question}': {str(e)}")

# Fixed usage example
def main_example():
    """
    Complete example of using the Advanced Multilingual Intent Classifier with robust error handling
    """
    
    print("🚀 Advanced Multilingual Intent Classifier Example")
    print("=" * 60)
    
    try:
        # Step 1: Initialize the classifier with your CSV file
        print("Step 1: Initializing classifier...")
        classifier = MultilingualIntentClassifier('questions_intents.csv')
        
        # Step 2: Load and preprocess data
        print("\nStep 2: Loading and preprocessing data...")
        data = classifier.load_and_preprocess_data()
        
        if len(data) == 0:
            raise ValueError("No data loaded. Please check your CSV file.")
        
        # Step 3: Prepare different feature extraction methods
        print("\nStep 3: Preparing feature extractors...")
        classifier.prepare_features()
        
        # Step 4: Split data into training and testing
        print("\nStep 4: Splitting data...")
        classifier.split_data(test_size=0.2, random_state=42)
        
        # Step 5: Train multiple models
        print("\nStep 5: Training multiple models...")
        best_combo, best_score = classifier.train_models()
        
        # Step 6: Create ensemble model (optional but recommended)
        print("\nStep 6: Creating ensemble model...")
        ensemble, ensemble_accuracy = classifier.create_ensemble_model()
        
        # Step 7: Detailed evaluation
        print("\nStep 7: Detailed evaluation...")
        classifier.evaluate_best_model()
        
        # Step 8: Test with multilingual examples
        print("\nStep 8: Testing with multilingual examples...")
        classifier.test_multilingual_examples()
        
        # Step 9: Save the trained model
        print("\nStep 9: Saving model...")
        classifier.save_model('advanced_multilingual_intent_classifier.pkl')
        
        print("\n🎉 Training completed successfully!")
        return classifier
        
    except Exception as e:
        print(f"❌ Error in main_example: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    classifier = main_example()