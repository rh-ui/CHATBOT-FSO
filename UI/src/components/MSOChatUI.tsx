import { useEffect, useState, useCallback, useRef } from 'react';
import { Minus, Mic, MicOff, Paperclip, Send } from 'lucide-react'
import '@/style/master.css';
import ROBOT from '@/assets/robot.png';
import CPU_AVATAR from '@/assets/Cpu.png';
import { Button } from './ui/button';
import { Input } from './ui/input';
import React, { Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';






/**
 *      COTE FRONTEND : bach nrj3 loading state dynamic
 * 
 *      Etapes : 
 *          - Etape 1 :
 *              1) remplacer fetch b EventSource for listening SSE
 *              2) Créer une connexion vers /search-stream
 *          - Etape 2 : 
 *              1) Ajouter un state pour le message de loading actuel
 *              2) update message in each received msg
 *              3) Afficher ce message dynamique au lieu du message statique
 *          - Etape 3 :  --> cycle de vie de la cnx
 *              1) ouvrir EventSource au debut de l'envoi
 *              2) listening les événements et mettre à jour l'UI
 *              3) fermer la connexion quand la reponse finale arrive
 *          - Etape 4 : 
 *              1) remplacer le msg : "je traite votre demande ..." par le msg dynamic
 *              3) ajouter des animations
 * 
 * 
 * 
 *        n.b: Types d'événements :
 *                -> 'status' → Appelle onStatusUpdate(message)
 *                -> 'final' → Résout la Promise avec les données
 *                -> 'error' → Rejette la Promisex²
 */




// Déclaration TypeScript pour la Web Speech API
type SpeechRecognition = typeof window.SpeechRecognition;
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: SpeechRecognition;
  }
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface Language {
  code: string;
  label: string;
  speechLang: string;
}

function Model() {
  const gltf = useGLTF('/models/file.glb');
  const ref = useRef<THREE.Object3D>(null);

  useFrame((state) => {
    const { mouse } = state;
    if (ref.current) {
      ref.current.rotation.y = mouse.x * Math.PI;
      ref.current.rotation.x = -mouse.y * Math.PI * 0.2;
    }
  });

  return <primitive ref={ref} object={gltf.scene} />;
}

useGLTF.preload('/models/file.glb');

function GLBViewer() {
  return (
    <Canvas camera={{ position: [0, 1, 3], fov: 45 }}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <Suspense fallback={null}>
        <Model />
      </Suspense>
      <OrbitControls enablePan={false} enableZoom={false} enableRotate={false} />
    </Canvas>
  );
}

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface ApiResponse {
  llm_used: boolean;
  structured_response?: string;
  confidence?: number;
  sources_used?: number;
  processing_time?: number;
  raw_results?: {
    question: string;
    answer: string;
    score: number;
    meta?: any;
  }[];
  results?: {
    question: string;
    answer: string;
    score: number;
    meta?: any;
  }[];
}

export default function MSOChatUI_test() {
  const useTypewriter = (text: string, speed: number = 50) => {
    const [displayText, setDisplayText] = useState('');
    const [isTyping, setIsTyping] = useState(false);

    useEffect(() => {
      if (!text) return;

      setIsTyping(true);
      setDisplayText('');

      let index = 0;
      const timer = setInterval(() => {
        setDisplayText(text.slice(0, index + 1));
        index++;

        if (index >= text.length) {
          clearInterval(timer);
          setIsTyping(false);
        }
      }, speed);

      return () => clearInterval(timer);
    }, [text, speed]);

    return { displayText, isTyping };
  };

  // Langues disponibles pour la reconnaissance vocale
  const availableLanguages: Language[] = [
    { code: 'fr', label: 'Français', speechLang: 'fr-FR' },
    { code: 'en', label: 'English', speechLang: 'en-US' },
    { code: 'ar', label: 'العربية', speechLang: 'ar-SA' }
  ];

  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // États pour la reconnaissance vocale
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<Language>(availableLanguages[0]);
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const quickActions = [
    "Doyen ?",
    "Comment m'inscrire ?",
    "Où voir les notes ?"
  ];

  // Fermer le dropdown quand on clique ailleurs
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowLanguageDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Vérification du support de la Web Speech API
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setSpeechSupported(true);
      recognitionRef.current = new SpeechRecognition();

      // Configuration de la reconnaissance vocale
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.maxAlternatives = 1;

      // Gestionnaires d'événements
      recognitionRef.current.onstart = () => {
        setIsListening(true);
        setSpeechError(null);
        setShowLanguageDropdown(false);
      };

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript;
        setMessage(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Erreur de reconnaissance vocale:', event.error);
        setIsListening(false);

        let errorMessage = '';
        switch (event.error) {
          case 'no-speech':
            errorMessage = 'Aucune parole détectée. Réessayez.';
            break;
          case 'audio-capture':
            errorMessage = 'Microphone non accessible.';
            break;
          case 'not-allowed':
            errorMessage = 'Permission microphone refusée.';
            break;
          case 'network':
            errorMessage = 'Erreur réseau lors de la reconnaissance.';
            break;
          default:
            errorMessage = 'Erreur de reconnaissance vocale.';
        }
        setSpeechError(errorMessage);

        // Effacer l'erreur après 3 secondes
        setTimeout(() => setSpeechError(null), 3000);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    } else {
      console.warn('Web Speech API non supportée par ce navigateur');
      setSpeechSupported(false);
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  // Mettre à jour la langue de reconnaissance quand elle change
  useEffect(() => {
    if (recognitionRef.current) {
      recognitionRef.current.lang = selectedLanguage.speechLang;
    }
  }, [selectedLanguage]);

  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      text: "👋 Bonjour et bienvenue sur le site de la Faculté des Sciences d'Oujda !\n\nJe suis M'SO, votre assistant virtuel. Posez-moi vos questions, je suis là pour vous aider en tous ce qui concerne la fso 📚",
      isUser: false,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const MessageWithTyping = ({ message, shouldType }: { message: Message, shouldType: boolean }) => {
    const { displayText, isTyping } = useTypewriter(shouldType ? message.text : '', 30);
    const textToShow = shouldType ? displayText : message.text;

    return (
      <span>
        {formatMessageText(textToShow)}
        {shouldType && isTyping && <span className="animate-pulse">|</span>}
      </span>
    );
  };

  const askQuestion = (question: string, lang = "fr", onStatusUpdate: (message: string) => void): Promise<ApiResponse> => {
    return new Promise((resolve, reject) => {
      fetch('http://localhost:8000/search-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, lang }),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body reader available');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          const readStream = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();

                if (done) {
                  break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                  if (line.trim() === '') continue;

                  if (line.startsWith('data: ')) {
                    try {
                      const jsonStr = line.substring(6);
                      const eventData = JSON.parse(jsonStr);

                      if (eventData.type === 'status') {
                        onStatusUpdate(eventData.message);
                      }
                      else if (eventData.type === 'final') {
                        reader.releaseLock();
                        resolve(eventData.data as ApiResponse);
                        return;
                      }
                      else if (eventData.type === 'error') {
                        reader.releaseLock();
                        reject(new Error(eventData.message));
                        return;
                      }
                    } catch (parseError) {
                      console.warn('Failed to parse SSE data:', line, parseError);
                    }
                  }
                }
              }

              reader.releaseLock();
              reject(new Error('Stream ended without final response'));

            } catch (streamError) {
              reader.releaseLock();
              reject(streamError);
            }
          };

          readStream();

        })
        .catch(fetchError => {
          reject(fetchError);
        });
    });
  };

  // Fonction pour gérer le clic sur le microphone
  const handleMicrophoneClick = useCallback(() => {
    if (!speechSupported) {
      setSpeechError('Reconnaissance vocale non supportée');
      return;
    }

    if (isListening) {
      // Arrêter l'écoute
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
      setShowLanguageDropdown(false);
    } else {
      // Afficher/masquer la liste des langues
      setShowLanguageDropdown(!showLanguageDropdown);
    }
  }, [isListening, speechSupported, showLanguageDropdown]);

  // Fonction pour démarrer la reconnaissance vocale avec une langue spécifique
  const startVoiceRecognition = useCallback((language: Language) => {
    if (!recognitionRef.current) return;

    setSelectedLanguage(language);
    setShowLanguageDropdown(false);

    try {
      recognitionRef.current.lang = language.speechLang;
      recognitionRef.current.start();
    } catch (error) {
      console.error('Erreur lors du démarrage de la reconnaissance:', error);
      setSpeechError('Erreur lors du démarrage de la reconnaissance vocale');
    }
  }, []);

  const handleSendMessage = async () => {
    if (message.trim() === '' || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: message,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    const tempResponse: Message = {
      id: (Date.now() + 1).toString(),
      text: "Initialisation...",
      isUser: false,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, tempResponse]);
    setTypingMessageId(tempResponse.id);

    const currentQuestion = message;
    setMessage('');
    setIsLoading(true);

    try {
      const response = await askQuestion(
        currentQuestion,
        selectedLanguage.code,
        (statusMessage: string) => {
          setMessages(prev => prev.map(msg =>
            msg.id === tempResponse.id
              ? {
                ...msg,
                text: statusMessage,
                timestamp: new Date()
              }
              : msg
          ));
          setTypingMessageId(tempResponse.id);
        }
      );

      const botResponses: Message[] = [];

      if (response.llm_used && response.structured_response) {
        let text = response.structured_response;
        botResponses.push({
          id: `${Date.now() + 2}`,
          text,
          isUser: false,
          timestamp: new Date()
        });
      } else if (response.results && response.results.length > 0) {
        botResponses.push(...response.results.map((result, index) => ({
          id: `${Date.now() + index + 2}`,
          text: `${result.question}\n\n${result.answer}`,
          isUser: false,
          timestamp: new Date()
        })));
      } else {
        botResponses.push({
          id: `${Date.now() + 2}`,
          text: "Désolé, je n'ai pas trouvé de réponse pertinente.",
          isUser: false,
          timestamp: new Date()
        });
      }

      setMessages(prev => [...prev.filter(msg => msg.id !== tempResponse.id), ...botResponses]);
      setTypingMessageId(tempResponse.id);

    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === tempResponse.id
          ? {
            ...msg,
            text: "Erreur de connexion avec le serveur",
            timestamp: new Date()
          }
          : msg
      ));
      setTypingMessageId(tempResponse.id);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isLoading) {
      handleSendMessage();
    }
  };

  const handleQuickAction = (action: string) => {
    setMessage(action);
  };

  const toggleChat = () => {
    setIsVisible(!isVisible);
  };

  function formatMessageText(text: string) {
    const parts = text.split(/(https?:\/\/[^\s]+)/g);
    return parts.map((part, i) =>
      part.match(/^https?:\/\/[^\s]+$/) ? (
        <a
          key={i}
          href={part}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 underline break-all"
        >
          {part}
        </a>
      ) : (
        <span key={i}>
          {part}
        </span>
      )
    );
  }

  return (
    <div className="relative">
      {/* Bouton pour afficher la fenêtre */}
      <div className="">
        <button
          onClick={toggleChat}
          className="fixed bottom-3 right-0 bg-transparent z-50"
        >
          <GLBViewer />
        </button>
      </div>

      {/* Chat */}
      {isVisible && (
        <div className="fixed bottom-30 right-10 w-[500px] h-[700px] bg-gray-50 rounded-3xl shadow-2xl border border-gray-200 flex flex-col z-40">
          {/* Header */}
          <div className="flex items-center justify-between p-5 pr-8 bg-transparent">
            <div className="flex items-center space-x-3">
              <div className="bg-transparent rounded-full flex items-center justify-center">
                <img src={ROBOT} alt="hh" width={49} height={57} />
              </div>
              <h2 className="text-xl font-normal text-gray-800">
                <span style={{ color: 'rgb(117, 168, 227)' }}>M'</span>SO
              </h2>
            </div>
            <button
              onClick={toggleChat}
              className="w-8 h-8 bg-transparent border-black border-1 rounded-full flex items-center justify-center hover:bg-gray-100 transition-colors"
            >
              <Minus className="w-5 h-5 text-black" />
            </button>
          </div>

          <hr />

          {/* Affichage des erreurs de reconnaissance vocale */}
          {speechError && (
            <div className="mx-4 mt-2 p-2 bg-red-100 border border-red-300 text-red-700 rounded-md text-sm">
              {speechError}
            </div>
          )}

          {/* Zone de chat */}
          <div className="flex-1 p-4 space-y-4 overflow-y-auto bg-gray-50 min-h-0">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.isUser ? 'justify-end' : 'items-start'} ${msg.isUser ? 'items-end space-x-2' : 'space-x-3'}`}>
                {!msg.isUser && (
                  <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <img src={CPU_AVATAR} alt="assistant" />
                  </div>
                )}
                <div className={`rounded-2xl p-4 max-w-xs shadow-sm ${msg.isUser
                  ? 'bg-blue-500 text-white rounded-tr-md'
                  : 'bg-white text-gray-800 rounded-tl-md'
                  }`}>
                  {msg.text.includes('\n\n') && !msg.isUser ? (
                    <>
                      <p className="font-semibold text-sm mb-2">
                        {msg.text.split('\n\n')[0]}
                      </p>
                      <p className="text-sm whitespace-pre-wrap">
                        {formatMessageText(msg.text.split('\n\n').slice(1).join('\n\n'))}
                      </p>
                    </>
                  ) : (
                    <p className={`text-sm leading-relaxed ${msg.isUser ? 'text-gray-50' : 'text-gray-800'}`}>
                      <MessageWithTyping
                        message={msg}
                        shouldType={msg.id === typingMessageId && !msg.isUser}
                      />
                    </p>
                  )}
                </div>
                {msg.isUser && (
                  <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-sm font-medium mb-1">
                    <span className='text-blue-600'>U</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Quick actions */}
          <div className="px-4 py-2 bg-white border-t border-gray-100 flex-shrink-0">
            <div className="flex gap-2 flex-wrap">
              {quickActions.map((action, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="text-xs px-3 py-2 rounded-full border-gray-200 hover:bg-gray-50 text-gray-600"
                  onClick={() => handleQuickAction(action)}
                >
                  {action}
                </Button>
              ))}
            </div>
          </div>

          {/* Zone d'entrée */}
          <div className="p-4 bg-transparent border-t border-gray-100 flex-shrink-0">
            <div className="relative">
              {/* Dropdown de sélection de langue */}
              {showLanguageDropdown && (
                <div
                  ref={dropdownRef}
                  className="absolute w-30 bottom-full mr-7 right-0 bg-white border border-gray-200 rounded-xl z-50 overflow-hidden"
                >
                  <div className="py-1">
                    {availableLanguages.map((language) => (
                      <button
                        key={language.code}
                        className={`w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors flex items-center justify-between ${selectedLanguage.code === language.code ? 'bg-blue-50 text-blue-500' : 'text-gray-700'
                          }`}
                        onClick={() => startVoiceRecognition(language)}
                      >
                        <span className="text-sm font-medium">{language.label}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-400">{language.code.toUpperCase()}</span>
                          {selectedLanguage.code === language.code && (
                            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center gap-3 bg-gray-50 rounded-full px-4 py-3 border border-gray-200">
                <Button variant="ghost" size="icon" className="text-gray-500 hover:text-gray-700">
                  <Paperclip className="w-5 h-5" />
                </Button>
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={isListening ? `Écoute en cours (${selectedLanguage.label})...` : "Posez votre question..."}
                  className="flex-1 border-none bg-transparent text-sm placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0"
                  disabled={isLoading}
                />

                {/* Bouton microphone avec indicateur de langue */}
                <div className="relative">
                  <Button
                    variant="ghost"
                    size="icon"
                    className={`transition-all duration-200 ${isListening
                      ? 'text-red-500 hover:text-red-600 bg-red-50'
                      : showLanguageDropdown
                        ? 'text-blue-500 hover:text-blue-400 bg-blue-50'
                        : speechSupported
                          ? 'text-gray-500 hover:text-gray-700'
                          : 'text-gray-300 cursor-not-allowed'
                      }`}
                    onClick={handleMicrophoneClick}
                    disabled={!speechSupported || isLoading}
                    title={
                      !speechSupported
                        ? 'Reconnaissance vocale non supportée'
                        : isListening
                          ? 'Cliquez pour arrêter l\'écoute'
                          : 'Cliquez pour choisir la langue et parler'
                    }
                  >
                    {isListening ? (
                      <MicOff className="w-5 h-5 animate-pulse" />
                    ) : (
                      <Mic className="w-5 h-5" />
                    )}
                  </Button>
                  {!isListening && (
                    <div className="absolute -bottom-1 -right-1 bg-orange-50 text-white text-xs px-1 rounded text-[10px] leading-tight">
                      {selectedLanguage.code.toUpperCase()}
                    </div>
                  )}
                </div>

                <Button
                  onClick={handleSendMessage}
                  size="icon"
                  className="bg-blue-500 hover:bg-blue-400 text-white rounded-full w-8 h-8"
                  disabled={isLoading || message.trim() === ''}
                >
                  {isLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}