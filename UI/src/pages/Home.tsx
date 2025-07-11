import Navbar from "@/components/Navbar";
import '@/style/master.css'
import act1 from "@/assets/act1.png";
import act2 from "@/assets/act2.png";
import act3 from "@/assets/act3.png";
import act4 from "@/assets/act4.png";
import DOYEN_IMG from "@/assets/doyen.png";
import GALLERY from "@/assets/gallery.png";
import { useState, useRef, useEffect } from "react";
import Footer from "@/components/footer";
import MSOChatUI from "@/components/MSOChatUI";
import logo from '/logo_fso.jpeg'

function Home() {

    function Actualites() {
        const actualitesData = [
            {
                id: 1,
                title: "ACTUALITÉ 1",
                description: "En savoir plus",
                image: act4,
                date: "01-13-2003"
            },
            {
                id: 2,
                title: "ACTUALITÉ 2",
                description: "En savoir plus",
                image: act3,
                date: "01-13-2003"
            },
            {
                id: 3,
                title: "ACTUALITÉ 3",
                description: "En savoir plus",
                image: act2,
                date: "01-13-2003"
            },
            {
                id: 4,
                title: "ACTUALITÉ 4",
                description: "En savoir plus",
                image: act1,
                date: "01-13-2003"
            }
        ];

        return (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 lg:gap-5 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-0">
                {actualitesData.map((item) => (
                    <div key={item.id} className="relative group">
                        <div className="aspect-[4/4] bg-gray-200 rounded-lg overflow-hidden shadow-lg">
                            <img
                                alt={item.title}
                                src={item.image}
                                className="w-full h-full object-cover"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
                            <div className="absolute bottom-1 left-3 right-3 sm:left-5 sm:right-5">
                                <h3 className="text-white font-bold text-sm sm:text-base lg:text-lg mb-1 p-1 uppercase tracking-wide">
                                    {item.title}
                                </h3>
                                <button className="bg-white/20 backdrop-blur-sm text-white px-3 py-1.5 sm:px-4 sm:py-2 rounded text-xs sm:text-sm hover:bg-white/30 transition-all duration-200 border border-white/30">
                                    {item.description}
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    function Info() {
        return (
            <section className="relative bg-img text-white pt-12 sm:pt-16 lg:pt-20 pb-16 sm:pb-24 lg:pb-32 min-h-[70vh] lg:min-h-[80vh]">
                <div className="absolute inset-0 bg-black/20"></div>
                <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10 h-full flex items-start">
                    <div className="max-w-4xl w-full mx-auto lg:mx-0 lg:ml-20 xl:ml-40 mt-8 sm:mt-16 lg:mt-40">
                        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-4 lg:mb-6 text-center lg:text-left">
                            FS <span className="text-orange-400">OUJDA</span>
                        </h1>
                        <p className="text-lg sm:text-xl lg:text-2xl mb-6 lg:mb-8 leading-relaxed text-center lg:text-left">
                            La Faculté des Sciences d'Oujda est un établissement public d'enseignement supérieur relevant de
                            l'Université Mohammed Premier, dédié à la formation, la recherche scientifique et l'innovation.
                        </p>
                        <div className="flex flex-wrap gap-4 mb-12 lg:mb-16 lg:mt-9 justify-center lg:justify-start xl:justify-center">
                            <button className="border-2 border-white text-white hover:bg-white hover:text-blue-900 px-6 py-3 lg:px-8 rounded-lg font-semibold transition-colors">
                                En savoir plus
                            </button>
                        </div>

                        <div className="mt-16 lg: xl:mt-74">
                            <div className="text-center mb-12 lg:mb-20 lg:ml-16 xl:ml-70">
                                <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-orange-400 mb-4">
                                    ACTUALITÉS
                                </h2>
                            </div>

                            <div className="w-full xl:ml-30">
                                <Actualites />
                            </div>

                            <div className="text-center mt-12 lg:mt-18 lg:ml-16 xl:ml-70">
                                <button className="bg-white text-blue-900 px-6 py-3 lg:px-8 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
                                    Voir plus d'actualités
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        );
    }

    function MotDoyen() {
        return (
            <section className="pt-12 sm:pt-16 lg:pt-20 px-4 sm:px-6 lg:px-8 lg:pl-30 mb-10">
                <div className="container mx-auto max-w-6xl">
                    <h1 className="text-blue-900 font-bold text-2xl sm:text-3xl lg:text-4xl mb-3">MOT DU DOYEN</h1>
                    <div className="w-20 sm:w-32 lg:w-80 h-1 lg:h-3 bg-orange-400"></div>

                    <div className="flex flex-col lg:flex-row gap-8 lg:gap-12 items-start mt-8 lg:mt-10">
                        <div className="w-full lg:w-1/3 flex justify-center lg:justify-start">
                            <img
                                src={DOYEN_IMG}
                                alt="doyen"
                                className="w-64 h-80 sm:w-72 sm:h-96 lg:w-80 lg:h-[453px] object-cover rounded-lg shadow-lg"
                            />
                        </div>
                        <div className="w-full lg:w-2/3">
                            <p className="text-base sm:text-lg lg:text-xl leading-relaxed lg:leading-10">
                                La Faculté des Sciences d'Oujda (FSO), créée le 18 avril 1979, est un pilier de l'Université Mohammed Premier (UMP). Forte de plus de 40 ans d'expérience, elle est un moteur de savoir, de recherche et d'innovation dans la région de l'Oriental.
                                Fruit des efforts collectifs de ses équipes, la FSO place l'étudiant au centre de son développement. Elle œuvre pour une formation de qualité, en lien avec les exigences du marché de l'emploi, tout en promouvant la recherche appliquée.
                                L'intégration du numérique, la culture entrepreneuriale et les soft/digital skills sont des axes clés pour accompagner les grandes mutations sociétales. Engagée, la FSO s'appuie sur une dynamique participative pour bâtir un avenir prometteur.
                                <br />— Prof. E. B. Maarouf, Doyen de la FSO
                            </p>
                        </div>
                    </div>
                </div>
            </section>
        );
    }

    function CountTotal() {
        const [isVisible, setIsVisible] = useState(false);
        const [animatedNumbers, setAnimatedNumbers] = useState([0, 0, 0]);
        const sectionRef = useRef(null);

        const statsData = [
            {
                number: 170,
                suffix: "+",
                label: "Enseignants"
            },
            {
                number: 15,
                suffix: "",
                label: "Département"
            },
            {
                number: 250,
                suffix: "+",
                label: "Étudiants inscrits"
            }
        ];

        useEffect(() => {
            const observer = new IntersectionObserver(
                ([entry]) => {
                    if (entry.isIntersecting) {
                        setIsVisible(true);
                    }
                },
                {
                    threshold: 0.3
                }
            );

            if (sectionRef.current) {
                observer.observe(sectionRef.current);
            }

            return () => {
                if (sectionRef.current) {
                    observer.unobserve(sectionRef.current);
                }
            };
        }, []);

        useEffect(() => {
            if (isVisible) {
                statsData.forEach((stat, index) => {
                    const duration = 2000;
                    const steps = 60;
                    const increment = stat.number / steps;
                    let currentStep = 0;

                    const timer = setInterval(() => {
                        currentStep++;
                        const currentValue = Math.min(Math.floor(increment * currentStep), stat.number);

                        setAnimatedNumbers(prev => {
                            const newNumbers = [...prev];
                            newNumbers[index] = currentValue;
                            return newNumbers;
                        });

                        if (currentStep >= steps) {
                            clearInterval(timer);
                        }
                    }, duration / steps);

                    return () => clearInterval(timer);
                });
            }
        }, [isVisible]);

        return (
            <section ref={sectionRef} className="bg-gray-50 py-16 lg:py-20">
                <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col sm:flex-row justify-center items-center gap-12 sm:gap-16 lg:gap-32">
                        {statsData.map((stat, index) => (
                            <div key={index} className="text-center">
                                <div className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold text-orange-400 mb-4">
                                    <span className="inline-block transition-all duration-300 ease-out">
                                        {animatedNumbers[index]}
                                    </span>
                                    {stat.suffix}
                                </div>
                                <div className="text-base sm:text-lg lg:text-xl text-orange-400 font-sm">
                                    {stat.label}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        );
    }

    function Gallerie() {
        return (
            <section className="pt-12 sm:pt-16 lg:pt-20 px-4 sm:px-6 lg:px-8 lg:pl-30 mb-10">
                <div className="container mx-auto max-w-6xl">
                    <h1 className="text-blue-900 font-bold text-2xl sm:text-3xl lg:text-4xl mb-3">GALLERY</h1>
                    <div className="w-20 sm:w-32 lg:w-80 h-1 lg:h-3 bg-orange-400"></div>

                    <div className="flex flex-col justify-center items-center">
                        <div className="pt-8 sm:pt-12 lg:pt-14 justify-center w-full">
                            <img
                                src={GALLERY}
                                alt="gal"
                                className="w-full h-48 sm:h-64 lg:h-96 xl:h-[530px] object-cover rounded-lg shadow-lg"
                            />
                        </div>

                        <div className="mt-8 sm:mt-10 lg:mt-12">
                            <button className="bg-orange-500 text-white px-6 py-3 lg:px-8 rounded-lg font-semibold hover:bg-orange-400 transition-colors">
                                Voir plus de photos
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        );
    }

    return (
        <>
            <Navbar />

            <main className="">
                <Info />
                <MotDoyen />
                <CountTotal />
                <Gallerie />
                <MSOChatUI />
            </main>

            <footer className="relative text-white mt-20 p-6 sm:p-8 lg:p-12 bg-footer">
                <Footer logo={logo} />
            </footer>
        </>
    );
}

export default Home;