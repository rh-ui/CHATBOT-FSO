import { useEffect, useState } from 'react';
import { Minus, Mic, Paperclip, ArrowUp, Send } from 'lucide-react';
import Spline from '@splinetool/react-spline';
import '@/style/master.css';
import ROBOT from '@/assets/robot.png';
import CPU_AVATAR from '@/assets/Cpu.png';
import { Button } from './ui/button';
import { Input } from './ui/input';


import React, { Suspense, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

import * as THREE from 'three';

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

// Preload the model
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
  results: {
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
  const [splineScene, setSplineScene] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const quickActions = [
    "Comment m'inscrire ?",
    "Où voir les notes ?",
    "FAQs"
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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          lang,
        }),
      });

      if (!response.ok) {
        throw new Error('Erreur lors de la requête');
      }

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

    // Réponse temporaire du bot
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

      const botResponses = response.results.length > 0
        ? response.results.map((result, index) => ({
          id: `${Date.now() + index + 2}`,
          text: `${result.question}\n\n${result.answer}`,
          isUser: false,
          timestamp: new Date()
        }))
        : [{
          id: `${Date.now() + 2}`,
          text: "Désolé, je n'ai pas trouvé de réponse pertinente.",
          isUser: false,
          timestamp: new Date()
        }];


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

  useEffect(() => {
    const loadScene = async () => {
      setSplineScene("https://prod.spline.design/l1R44eHGk3Qq8SAI/scene.splinecode");
    };
    loadScene();
  }, []);

  const toggleChat = () => {
    setIsVisible(!isVisible);
  };

  return (
    <div className="relative ">
      {/* AI Icon Button */}
      <div className="">
        <button
          onClick={toggleChat}
          className="fixed bottom-3 right-0 bg-transparent z-50  "
        >
          {/* {splineScene && <Spline scene={splineScene} className='' />} */}
          <GLBViewer />
        </button>
      </div>

      {/* Chat Interface */}
      {isVisible && (
        <div className="fixed bottom-30 right-10 w-[400px] h-[600px] bg-gray-50 rounded-3xl shadow-2xl border border-gray-200 flex flex-col z-40">
          {/* Header */}
          <div className="flex items-center justify-between p-5 pr-8 bg-transparent">
            <div className="flex items-center space-x-3">
              {/* Robot Avatar */}
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

          {/* Chat Content */}
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
                      <p className="text-sm">
                        {msg.text.split('\n\n').slice(1).join('\n\n')}
                      </p>
                    </>
                  ) : (
                    <p className={`text-sm leading-relaxed ${msg.isUser ? 'text-gray-50' : 'text-gray-800'}`}>
                      {msg.text.split('\n').map((line, index) => (
                        <span key={index}>
                          {line}
                          {index < msg.text.split('\n').length - 1 && <br />}
                        </span>
                      ))}
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

          {/* Quick Actions */}
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

          {/* Input Area */}
          <div className="p-4 bg-transparent border-t border-gray-100 flex-shrink-0">
            <div className="flex items-center gap-3 bg-gray-50 rounded-full px-4 py-3 border border-gray-200">
              <Button variant="ghost" size="icon" className="text-gray-500 hover:text-gray-700">
                <Paperclip className="w-5 h-5" />
              </Button>

              <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Posez votre question..."
                className="flex-1 border-none bg-transparent text-sm placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0"
                disabled={isLoading}
              />

              <Button variant="ghost" size="icon" className="text-gray-500 hover:text-gray-700">
                <Mic className="w-5 h-5" />
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
        </div>
      )}
    </div>
  );
}