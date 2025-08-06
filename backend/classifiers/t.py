import spacy
from spacy.matcher import PhraseMatcher
from langdetect import detect

class QuestionSimplifier:
    def __init__(self):
        # Load multilingual models
        self.nlp_fr = spacy.load("fr_core_news_md")
        self.nlp_en = spacy.load("en_core_web_md")
        
        # Domain-specific patterns (can be expanded)
        self.domain_terms = {
            'fr': ['doyen', 'étudiant', 'inscription', 'nouveau', 'bourse'],
            'en': ['dean', 'student', 'registration', 'new', 'scholarship'],
            'ar': ['عميد', 'طالب', 'تسجيل']  # Arabic examples
        }
        
        # Initialize matchers
        self.matchers = {
            'fr': self._create_matcher(self.nlp_fr, 'fr'),
            'en': self._create_matcher(self.nlp_en, 'en')
        }
    
    def _create_matcher(self, nlp, lang):
        matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp(text) for text in self.domain_terms[lang]]
        matcher.add("DOMAIN_TERMS", patterns)
        return matcher
    
    def simplify(self, question):
        # Step 1: Language detection
        lang = detect(question)[:2]  # Get 2-letter code
        
        # Step 2: Process with appropriate NLP pipeline
        if lang == 'fr':
            doc = self.nlp_fr(question)
            matcher = self.matchers['fr']
        else:  # default to English
            doc = self.nlp_en(question)
            matcher = self.matchers['en']
        
        # Step 3: Extract key information
        results = set()
        
        # A. Get domain terms via pattern matching
        matches = matcher(doc)
        for match_id, start, end in matches:
            results.add(doc[start:end].lemma_)
        
        # B. Get content words via POS tagging
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                results.add(token.lemma_)
        
        return ' '.join(sorted(results))


simplifier = QuestionSimplifier()

