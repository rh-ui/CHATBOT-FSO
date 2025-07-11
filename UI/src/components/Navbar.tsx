import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu"
import { cn } from "@/lib/utils"
import { Menu, X } from "lucide-react";
import { useState } from "react";

const navigationItems = [
  { title: "Facultés", href: "#" },
  { title: "Formations", href: "#" },
  { title: "Enseignants", href: "#" },
  { title: "Espace étudiant", href: "#" },
  { title: "Actualités", href: "#" },
  { title: "Contact", href: "#" },
]

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background">
      <div className="container mx-auto px-4 py-3 lg:py-4">
        <div className="flex items-center justify-between">
          {/* Logo et titre */}
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 sm:w-16 sm:h-16 lg:w-20 lg:h-20">
              <img
                src="/logo_fso.jpeg?height=202&width=94"
                alt="Logo Université Mohammed Premier"
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* Menu mobile button */}
          <button
            className="lg:hidden text-gray-500 hover:text-gray-700 p-2"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>

          {/* Desktop Navigation */}
          <NavigationMenu className="hidden lg:flex">
            <NavigationMenuList>
              {navigationItems.map((item) => (
                <NavigationMenuItem key={item.title}>
                  <NavigationMenuLink
                    href={item.href}
                    className={cn(
                      "group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50",
                      "text-muted-foreground hover:text-orange-500",
                    )}
                  >
                    {item.title}
                  </NavigationMenuLink>
                </NavigationMenuItem>
              ))}
              <NavigationMenuItem>
                <NavigationMenuLink
                  href="#"
                  className={cn(
                    "group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50",
                    "text-muted-foreground hover:text-orange-500 border-l border-border ml-4 pl-4",
                  )}
                >
                  FR
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="lg:hidden mt-4 py-4 border-t border-gray-200">
            <nav>
              <ul className="space-y-3">
                {navigationItems.map((item) => (
                  <li key={item.title}>
                    <a
                      href={item.href}
                      className="block text-gray-600 hover:text-orange-500 transition-colors duration-200 font-medium py-2"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      {item.title}
                    </a>
                  </li>
                ))}
                <li className="border-t border-gray-200 pt-3">
                  <a
                    href="#"
                    className="block text-gray-600 hover:text-orange-500 transition-colors duration-200 font-medium py-2"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    FR
                  </a>
                </li>
              </ul>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}