import { useEffect, useState, Suspense, useRef } from 'react';
import { Minus, Mic, Paperclip, ArrowUp, Send } from 'lucide-react';
import '@/style/master.css';
import ROBOT from '@/assets/robot.png';
import CPU_AVATAR from '@/assets/Cpu.png';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { MicOff, Upload, X } from 'lucide-react';

function Model() {
  const gltf = useGLTF('/models/file.glb');
  const ref = useRef<THREE.Object3D>(null);

  useFrame((state) => {
    const { mouse } = state;
    // Make model rotate slowly based on mouse X/Y position
    if (ref.current) {
      ref.current.rotation.y = mouse.x * Math.PI;    // left/right
      ref.current.rotation.x = -mouse.y * Math.PI * 0.2; // up/down, smaller effect
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
  isAudio?: boolean;
  // transcription?: string;
  transcription?: {
    original_text: string;
    detected_language: string;
    duration: number;
    filename: string;
  };
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

export default function MSOChatUI() {
  const [isVisible, setIsVisible] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [showFileUpload, setShowFileUpload] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const quickActions = [
    "Qui est le Doyen ?",
    "Comment m'inscrire ?",
    "Où voir les notes ?"
  ];

  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      text: "👋 Bonjour et bienvenue sur le site de la Faculté des Sciences d'Oujda !\n\nJe suis M'SO, votre assistant virtuel. Posez-moi vos questions, je suis là pour vous aider en tous ce qui concerne la fso 📚",
      isUser: false,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const askQuestion = async (question: string, lang = "fr") => {
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, lang }),
      });

      if (!response.ok) throw new Error('Erreur lors de la requête');

      return await response.json() as ApiResponse;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  };

  const handleSendMessage = async () => {
    if (message.trim() === '' || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: message,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setMessage('');

    const tempResponse: Message = {
      id: (Date.now() + 1).toString(),
      text: "Je traite votre demande...",
      isUser: false,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, tempResponse]);

    setIsLoading(true);
    try {
      const response = await askQuestion(message);

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

  useEffect(() => {
    return () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    };
  }, [mediaRecorder]);


  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Réinitialiser les chunks AVANT de démarrer l'enregistrement
      setAudioChunks([]);

      const options = {
        mimeType: getSupportedMimeType(),
        audioBitsPerSecond: 128000
      };

      const recorder = new MediaRecorder(stream, options);

      let localAudioChunks: Blob[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          localAudioChunks.push(event.data);
          setAudioChunks(prev => [...prev, event.data]);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());

        const mimeType = getSupportedMimeType();
        const audioBlob = new Blob(localAudioChunks, { type: mimeType });
        const extension = getExtensionFromMimeType(mimeType);
        const audioFile = new File([audioBlob], `recording_${Date.now()}.${extension}`, {
          type: mimeType
        });

        console.log(`Created audio file: ${audioFile.name}, size: ${audioFile.size} bytes, type: ${audioFile.type}`);
        handleAudioMessage(audioFile);

        // Nettoyer
        localAudioChunks = [];
        setAudioChunks([]);
      };

      setMediaRecorder(recorder);
      recorder.start(1000);
      setIsRecording(true);
    } catch (error) {
      console.error('Erreur microphone:', error);
      alert('Impossible d\'accéder au microphone. Vérifiez les permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  // Helper function to get supported MIME type
  const getSupportedMimeType = () => {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/wav'
    ];

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return 'audio/wav';
  };

  const getExtensionFromMimeType = (mimeType: string) => {
    const mimeMap: Record<string, string> = {
      'audio/webm;codecs=opus': 'webm',
      'audio/webm': 'webm',
      'audio/mp4': 'mp4',
      'audio/wav': 'wav',
      'audio/mpeg': 'mp3',
      'audio/ogg': 'ogg'
    };

    return mimeMap[mimeType] || 'webm';
  };


  // const stopRecording = () => {
  //   if (mediaRecorder && mediaRecorder.state === 'recording') {
  //     mediaRecorder.stop();
  //     setIsRecording(false);

  //     mediaRecorder.onstop = () => {
  //       const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
  //       const audioFile = new File([audioBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
  //       handleAudioMessage(audioFile);
  //     };
  //   }
  // };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const supportedFormats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

      console.log(`Uploaded file: ${file.name}, size: ${file.size} bytes, type: ${file.type}`);

      if (supportedFormats.includes(fileExtension)) {
        // Validate file size (max 25MB)
        if (file.size > 25 * 1024 * 1024) {
          alert('Le fichier est trop volumineux (max 25MB)');
          return;
        }

        handleAudioMessage(file);
        setShowFileUpload(false);
      } else {
        alert(`Format non supporté. Formats acceptés: ${supportedFormats.join(', ')}`);
      }
    }
    event.target.value = '';
  };

  const handleAudioMessage = async (audioFile: File) => {

    if (isLoading) return;

    console.log(`Processing audio file: ${audioFile.name}, size: ${audioFile.size}, type: ${audioFile.type}`);

    if (audioFile.size === 0) {
      alert('Le fichier audio est vide');
      return;
    }

    // Message temporaire
    const tempResponse: Message = {
      id: (Date.now() + 1).toString(),
      text: "Transcription en cours...",
      isUser: false,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, tempResponse]);

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('audio_file', audioFile);
      formData.append('lang', 'fr');
      formData.append('use_llm', 'true');

      console.log('Sending audio to server...');

      const response = await fetch('http://localhost:8000/voice-search', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('Server response:', result);

      const botResponses: Message[] = [];

      // Ajouter transcription
      if (result.transcription?.original_text) {
        const transcriptionMessage: Message = {
          id: `${Date.now() + 100}`,
          text: result.transcription.original_text,
          isUser: true,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, transcriptionMessage]);
      }

      // Ajouter réponse LLM
      if (result.structured_response) {
        botResponses.push({
          id: `${Date.now() + 3}`,
          text: result.structured_response,
          isUser: false,
          timestamp: new Date()
        });
      } else if (result.results && result.results.length > 0) {
        botResponses.push(...result.results.map((res: any, index: number) => ({
          id: `${Date.now() + index + 3}`,
          text: `${res.question}\n\n${res.answer}`,
          isUser: false,
          timestamp: new Date()
        })));
      }

      if (botResponses.length === 0) {
        botResponses.push({
          id: `${Date.now() + 2}`,
          text: "Désolé, je n'ai pas pu traiter votre audio correctement.",
          isUser: false,
          timestamp: new Date()
        });
      }

      setMessages(prev => [...prev.filter(msg => msg.id !== tempResponse.id), ...botResponses]);

    } catch (error) {
      console.error('Audio processing error:', error);

      // Gérer l'erreur avec plus de détails
      const errorMessage = error instanceof Error ? error.message : 'Erreur inconnue';

      setMessages(prev => prev.map(msg =>
        msg.id === tempResponse.id
          ? { ...msg, text: `Erreur audio: ${errorMessage}` }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

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
                  } ${msg.isAudio ? 'border-none' : ''} `}>
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
                      {formatMessageText(msg.text)}
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

          {/* Quick actions (les questions li frequent) */}
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
          <div className="p-4 bg-gray-50 rounded-full border-t border-gray-100 flex-shrink-0">
            {isRecording && (
              <div className="flex items-center justify-center mt-2 mb-3">
                <div className="flex items-center gap-2 text-red-500 text-sm">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                  Enregistrement en cours...
                </div>
              </div>
            )}
            <div className="flex items-center gap-3 bg-white rounded-full px-4 py-3 border border-gray-200">
              <Button variant="ghost" size="icon" className="text-gray-500 hover:text-gray-700">
                <Paperclip className="w-5 h-5" /> {/* add file button */}
              </Button>
              <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Posez votre question..."
                className="flex-1 border-none bg-transparent text-sm placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0"
                disabled={isLoading}
              />
              <Button
                variant="ghost"
                size="icon"
                className={`${isRecording ? 'text-red-500 bg-red-50' : 'text-gray-500'}`}
                onClick={isRecording ? stopRecording : startRecording}
              >
                {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </Button>
              <Button
                onClick={handleSendMessage}
                size="icon"
                className="bg-blue-500 hover:bg-blue-600 text-white rounded-full w-8 h-8"
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>


          </div>

          {/* Upload de fichier audio */}
          {showFileUpload && (
            <div className="px-4 py-2 bg-purple-50 border-t border-purple-200 flex-shrink-0">
              <div className="flex items-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp3,.wav,.m4a,.flac,.ogg,.webm,.mp4"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Button onClick={() => fileInputRef.current?.click()} variant="outline" size="sm" className="flex-1">
                  <Upload className="w-4 h-4 mr-2" />
                  Choisir un fichier audio
                </Button>
                <Button onClick={() => setShowFileUpload(false)} variant="ghost" size="sm">
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
