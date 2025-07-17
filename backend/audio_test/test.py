# from transformers import pipeline
# import torch
# import time
# import soundfile as sf
# import numpy as np

# # 1. Configuration
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device : {device}")

# # 2. Chargement du modèle (version base pour meilleure détection)
# model_name = "openai/whisper-base"
# stt_pipeline = pipeline(
#     "automatic-speech-recognition",
#     model=model_name,
#     device=device
# )

# # 3. Fonction de traitement audio
# def process_audio(file_path):
#     try:
#         # Lecture du fichier
#         audio_data, sample_rate = sf.read(file_path)
        
#         # Conversion mono et 16kHz si nécessaire
#         if len(audio_data.shape) > 1:
#             audio_data = np.mean(audio_data, axis=1)
#         if sample_rate != 16000:
#             import librosa
#             audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
#         return {"raw": audio_data, "sampling_rate": 16000}
    
#     except Exception as e:
#         raise RuntimeError(f"Erreur traitement audio: {str(e)}")

# # 4. Transcription avec détection automatique
# audio_file = "arabic_test.wav"
# try:
#     print("\nDébut analyse...")
#     start_time = time.time()
    
#     # Traitement audio
#     audio_input = process_audio(audio_file)
    
#     # Transcription avec détection automatique de langue
#     # Whisper détecte automatiquement la langue sans paramètre spécial
#     transcript_result = stt_pipeline(
#         audio_input,
#         return_timestamps=True,
#         generate_kwargs={
#             "task": "transcribe"
#         }
#     )
    
#     print(f"\nTexte transcrit : {transcript_result['text']}")
#     print(f"Temps total : {time.time() - start_time:.2f}s")
    
#     # Alternative: Si vous voulez forcer une langue spécifique
#     # Décommentez les lignes suivantes pour forcer le français
#     """
#     transcript_result_fr = stt_pipeline(
#         audio_input,
#         return_timestamps=False,
#         generate_kwargs={
#             "task": "transcribe",
#             "language": "french"
#         }
#     )
#     print(f"\nTexte transcrit (français forcé) : {transcript_result_fr['text']}")
#     """

# except Exception as e:
#     print(f"\n❌ Erreur : {str(e)}")

# # 5. Version alternative avec détection manuelle de langue
# def detect_language_manual(audio_input):
#     """
#     Détection manuelle de langue en testant différentes langues
#     et en choisissant celle avec le meilleur score de confiance
#     """
#     languages = ["french", "english", "spanish", "german", "italian"]
#     results = {}
    
#     for lang in languages:
#         try:
#             result = stt_pipeline(
#                 audio_input,
#                 return_timestamps=False,
#                 generate_kwargs={
#                     "task": "transcribe",
#                     "language": lang
#                 }
#             )
#             # On pourrait utiliser un score de confiance ici si disponible
#             results[lang] = result['text']
#         except:
#             continue
    
#     return results

# # Exemple d'utilisation de la détection manuelle
# print("\n" + "="*50)
# print("DÉTECTION MANUELLE DE LANGUE")
# print("="*50)

# try:
#     audio_input = process_audio(audio_file)
#     lang_results = detect_language_manual(audio_input)
    
#     print("\nRésultats par langue:")
#     for lang, text in lang_results.items():
#         print(f"{lang.capitalize()}: {text}")
        
# except Exception as e:
#     print(f"Erreur détection manuelle: {str(e)}")



import speech_recognition as sr
import whisper
import torch
import numpy as np
import wave
import tempfile
import os
import logging
import time
from pydub import AudioSegment
from threading import Thread
import queue

class GPUMultilingualSTT:
    def __init__(self, model_size="base", device=None):
        """
        Initialiser le système STT avec support GPU
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
            device: "cuda", "cpu" ou None pour auto-détection
        """
        
        # Configuration du device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Utilisation du device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
            print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Charger le modèle Whisper sur GPU
        print(f"Chargement du modèle Whisper '{model_size}' sur {self.device}...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        # Optimisations GPU
        if self.device == "cuda":
            # Activer les optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Précharger le modèle
            self._warmup_model()
        
        # Initialiser SpeechRecognition
        self.recognizer = sr.Recognizer()
        
        # Configuration microphone
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"Microphone non disponible: {e}")
            self.microphone = None
        
        # Queue pour traitement en parallèle
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Démarrer le worker thread
        self.processing_thread = Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
    
    def _warmup_model(self):
        """Préchauffer le modèle GPU avec un audio factice"""
        try:
            print("Préchauffage du modèle GPU...")
            # Créer un audio factice de 1 seconde
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            # Mesurer le temps de la première inférence
            start_time = time.time()
            self.whisper_model.transcribe(dummy_audio, verbose=False)
            warmup_time = time.time() - start_time
            
            print(f"Modèle préchauffé en {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"Erreur lors du préchauffage: {e}")
    
    def _processing_worker(self):
        """Worker thread pour traitement en parallèle"""
        while True:
            try:
                task = self.processing_queue.get()
                if task is None:
                    break
                
                audio_data, options = task
                result = self._process_audio_gpu(audio_data, options)
                self.results_queue.put(result)
                
            except Exception as e:
                logging.error(f"Erreur dans le worker: {e}")
                self.results_queue.put(None)
    
    def get_optimal_model_size(self):
        """Recommander la taille de modèle selon la mémoire GPU"""
        if self.device == "cpu":
            return "base"
        
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 10:
                return "large"
            elif gpu_memory >= 6:
                return "medium" 
            elif gpu_memory >= 4:
                return "small"
            else:
                return "base"
                
        except Exception:
            return "base"
    
    def monitor_gpu_usage(self):
        """Monitorer l'utilisation GPU"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'utilization': f"{allocated:.1f}GB allouée, {cached:.1f}GB en cache"
            }
        return None
    
    def convert_audio_for_gpu(self, audio_path):
        """Convertir et optimiser l'audio pour traitement GPU"""
        try:
            # Charger avec pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Convertir en format optimal pour Whisper
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Convertir en numpy array
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / 32768.0  # Normaliser
            
            return audio_data
            
        except Exception as e:
            logging.error(f"Erreur conversion audio: {e}")
            return None
    
    def _process_audio_gpu(self, audio_data, options):
        """Traitement audio optimisé GPU"""
        try:
            start_time = time.time()
            
            # Traitement avec Whisper sur GPU
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                result = self.whisper_model.transcribe(
                    audio_data,
                    language=options.get('language'),
                    task=options.get('task', 'transcribe'),
                    beam_size=options.get('beam_size', 5),
                    best_of=options.get('best_of', 5),
                    temperature=options.get('temperature', 0.0),
                    compression_ratio_threshold=options.get('compression_ratio_threshold', 2.4),
                    logprob_threshold=options.get('logprob_threshold', -1.0),
                    no_speech_threshold=options.get('no_speech_threshold', 0.6),
                    verbose=False
                )
            
            processing_time = time.time() - start_time
            
            # Calculer les métriques de performance
            audio_duration = len(audio_data) / 16000
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'confidence': self._calculate_confidence(result),
                'processing_time': processing_time,
                'audio_duration': audio_duration,
                'rtf': rtf,  # Real-time factor
                'segments': result.get('segments', []),
                'method': f'whisper_gpu_{self.device}'
            }
            
        except Exception as e:
            logging.error(f"Erreur traitement GPU: {e}")
            return None
    
    def _calculate_confidence(self, whisper_result):
        """Calculer un score de confiance basé sur les segments"""
        try:
            segments = whisper_result.get('segments', [])
            if not segments:
                return 0.8
            
            # Moyenner les probabilités des segments
            total_prob = 0
            total_length = 0
            
            for segment in segments:
                if 'avg_logprob' in segment:
                    # Convertir log prob en probabilité
                    prob = np.exp(segment['avg_logprob'])
                    length = segment['end'] - segment['start']
                    total_prob += prob * length
                    total_length += length
            
            if total_length > 0:
                return min(total_prob / total_length, 1.0)
            else:
                return 0.8
                
        except Exception:
            return 0.8
    
    def transcribe_batch(self, audio_paths, options=None):
        """Transcription en lot pour efficacité GPU"""
        if options is None:
            options = {}
        
        results