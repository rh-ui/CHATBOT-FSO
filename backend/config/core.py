
from pathlib import Path

CLASSIFIER_MODEL = Path(__file__).parent.parent / 'classifiers' / 'advanced_multilingual_intent_classifier.pkl'
DATASET = Path(__file__).parent.parent / 'data' / 'dataset_dict_date.json'



LANG_MAP = {
    'fr': 'fr',
    'en': 'en',
    'ar': 'ar',    
    'amz': 'amz'
}