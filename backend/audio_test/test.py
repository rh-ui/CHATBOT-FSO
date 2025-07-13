from transformers import pipeline
import torch
import time
import soundfile as sf
import numpy as np

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# 2. Chargement du modèle (version base pour meilleure détection)
model_name = "openai/whisper-base"
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    device=device
)

# 3. Fonction de traitement audio
def process_audio(file_path):
    try:
        # Lecture du fichier
        audio_data, sample_rate = sf.read(file_path)
        
        # Conversion mono et 16kHz si nécessaire
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        return {"raw": audio_data, "sampling_rate": 16000}
    
    except Exception as e:
        raise RuntimeError(f"Erreur traitement audio: {str(e)}")

# 4. Transcription avec détection automatique
audio_file = "arabic_test.wav"
try:
    print("\nDébut analyse...")
    start_time = time.time()
    
    # Traitement audio
    audio_input = process_audio(audio_file)
    
    # Transcription avec détection automatique de langue
    # Whisper détecte automatiquement la langue sans paramètre spécial
    transcript_result = stt_pipeline(
        audio_input,
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe"
        }
    )
    
    print(f"\nTexte transcrit : {transcript_result['text']}")
    print(f"Temps total : {time.time() - start_time:.2f}s")
    
    # Alternative: Si vous voulez forcer une langue spécifique
    # Décommentez les lignes suivantes pour forcer le français
    """
    transcript_result_fr = stt_pipeline(
        audio_input,
        return_timestamps=False,
        generate_kwargs={
            "task": "transcribe",
            "language": "french"
        }
    )
    print(f"\nTexte transcrit (français forcé) : {transcript_result_fr['text']}")
    """

except Exception as e:
    print(f"\n❌ Erreur : {str(e)}")

# 5. Version alternative avec détection manuelle de langue
def detect_language_manual(audio_input):
    """
    Détection manuelle de langue en testant différentes langues
    et en choisissant celle avec le meilleur score de confiance
    """
    languages = ["french", "english", "spanish", "german", "italian"]
    results = {}
    
    for lang in languages:
        try:
            result = stt_pipeline(
                audio_input,
                return_timestamps=False,
                generate_kwargs={
                    "task": "transcribe",
                    "language": lang
                }
            )
            # On pourrait utiliser un score de confiance ici si disponible
            results[lang] = result['text']
        except:
            continue
    
    return results

# Exemple d'utilisation de la détection manuelle
print("\n" + "="*50)
print("DÉTECTION MANUELLE DE LANGUE")
print("="*50)

try:
    audio_input = process_audio(audio_file)
    lang_results = detect_language_manual(audio_input)
    
    print("\nRésultats par langue:")
    for lang, text in lang_results.items():
        print(f"{lang.capitalize()}: {text}")
        
except Exception as e:
    print(f"Erreur détection manuelle: {str(e)}")