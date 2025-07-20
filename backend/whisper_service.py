import whisper
import torch
import tempfile
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper utilise le device: {self.device}")
        
        try:
            self.model = whisper.load_model("small", device=self.device)
            logger.info("Modèle Whisper 'small' chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Whisper: {e}")
            try:
                self.model = whisper.load_model("base", device=self.device)
                logger.info("Modèle Whisper 'base' chargé en fallback")
            except Exception as e2:
                logger.error(f"Erreur critique Whisper: {e2}")
                raise
        
        self.transcribe_options = {
            "fp16": True if self.device == "cuda" else False,  # Précision 16-bit pour CUDA
            "language": None,  # Auto-détection de langue
            "task": "transcribe",
            "verbose": False
        }
    
    def transcribe_audio_file(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcrit un fichier audio"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
            
            options = self.transcribe_options.copy()
            if language:
                options["language"] = language
            
            logger.info(f"Transcription en cours de: {audio_path}")
            
            # Transcription
            result = self.model.transcribe(audio_path, **options)
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "duration": sum(seg.get("end", 0) - seg.get("start", 0) for seg in result.get("segments", []))
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    def transcribe_audio_data(self, audio_data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """Transcrit des données audio (bytes) en utilisant un fichier temporaire"""
        try:
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Transcrire le fichier temporaire
                result = self.transcribe_audio_file(temp_path)
                return result
            finally:
                # Nettoyer le fichier temporaire
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Erreur lors de la transcription des données audio: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    def get_supported_formats(self) -> list:
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "small" if hasattr(self.model, 'dims') else "unknown",
            "device": self.device,
            "supported_languages": whisper.tokenizer.TO_LANGUAGE_CODE.keys() if hasattr(whisper, 'tokenizer') else [],
            "supported_formats": self.get_supported_formats()
        }

whisper_service = WhisperService()


