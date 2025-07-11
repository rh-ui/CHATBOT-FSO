import { Facebook, Instagram, Linkedin } from 'lucide-react';
import '@/style/master.css'


const Footer = ({ logo }) => {
    return (
        <div className="relative z-10 max-w-7xl mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
                {/* Logo and Contact Info Section */}
                <div className="flex flex-col md:flex-row items-start md:items-center space-y-4 md:space-y-0 md:space-x-8">
                    {/* Logo */}
                    <div className="">
                        {logo ? (
                            <img
                                src={logo}
                                alt="University Logo"
                                className="p-4"
                                width={194}
                                height={170}
                            />
                        ) : (
                            <div className="w-16 h-16 md:w-20 md:h-20 bg-orange-600 flex items-center justify-center">
                                <div className="text-white text-xl font-bold">FSO</div>
                            </div>
                        )}
                    </div>

                    {/* Contact Information */}
                    <div className="space-y-2 text-lg leading-10 ml-30">
                        <div className="font-medium">
                            Adresse : Faculté des Sciences BV Mohammed VI - BP 717
                        </div>
                        <div>
                            Oujda 60000 Maroc
                        </div>
                        <div>
                            Phone : +212 536500601/2
                        </div>
                        <div>
                            Fax : +212 536500603
                        </div>
                    </div>
                </div>

                {/* Social Media Icons */}

            </div>

            {/* Copyright Section */}
            <div className="mt-8 pt-6 border-t border-slate-700 text-center text-sm flex flex-row justify-between h-">
                <p className='text-lg'>Tous droits réservés ump.ma - © 2025</p>
                <div className="flex space-x-4 mt-6 md:mt-0">
                    <a
                        href="#"
                        className="text-white hover:text-blue-400 transition-colors duration-300"
                        aria-label="Facebook"
                    >
                        <Facebook size={30} />
                    </a>
                    <a
                        href="#"
                        className="text-white hover:text-pink-400 transition-colors duration-300"
                        aria-label="Instagram"
                    >
                        <Instagram size={30} />
                    </a>
                    <a
                        href="#"
                        className="text-white hover:text-blue-300 transition-colors duration-300"
                        aria-label="LinkedIn"
                    >
                        <Linkedin size={30} />
                    </a>
                </div>
            </div>
        </div>
    );
};

export default Footer;