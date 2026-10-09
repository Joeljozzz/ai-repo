import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { Brain } from "lucide-react";
import "./globals.css";
import { curriculum } from "@/data/curriculum";

import * as LucideIcons from "lucide-react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Engineer's Handbook",
  description: "Comprehensive PhD-level guide to ML, Deep Learning, and AI Architectures",
};

const colorMap: Record<string, { text: string; bg: string; border: string; hoverText: string; hoverBg: string }> = {
  blue: { text: "text-blue-500", bg: "bg-blue-100", border: "border-blue-100", hoverText: "hover:text-blue-600", hoverBg: "hover:bg-blue-50" },
  emerald: { text: "text-emerald-500", bg: "bg-emerald-100", border: "border-emerald-100", hoverText: "hover:text-emerald-600", hoverBg: "hover:bg-emerald-50" },
  teal: { text: "text-teal-500", bg: "bg-teal-100", border: "border-teal-100", hoverText: "hover:text-teal-600", hoverBg: "hover:bg-teal-50" },
  purple: { text: "text-purple-500", bg: "bg-purple-100", border: "border-purple-100", hoverText: "hover:text-purple-600", hoverBg: "hover:bg-purple-50" },
  indigo: { text: "text-indigo-500", bg: "bg-indigo-100", border: "border-indigo-100", hoverText: "hover:text-indigo-600", hoverBg: "hover:bg-indigo-50" },
  rose: { text: "text-rose-500", bg: "bg-rose-100", border: "border-rose-100", hoverText: "hover:text-rose-600", hoverBg: "hover:bg-rose-50" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} bg-gradient-to-br from-indigo-300 via-purple-300 to-pink-300 text-slate-900 flex h-screen w-screen items-center justify-center p-4 sm:p-8 overflow-hidden`}>
        
        <div className="w-full h-full max-w-7xl max-h-[900px] flex flex-col bg-white rounded-xl shadow-2xl overflow-hidden ring-1 ring-slate-900/10">
          
          <div className="h-12 bg-slate-100 border-b border-slate-200 flex items-center px-4 shrink-0 justify-between select-none">
            <div className="flex space-x-2 w-20">
              <div className="w-3.5 h-3.5 rounded-full bg-red-500 border border-red-600 shadow-inner"></div>
              <div className="w-3.5 h-3.5 rounded-full bg-amber-500 border border-amber-600 shadow-inner"></div>
              <div className="w-3.5 h-3.5 rounded-full bg-emerald-500 border border-emerald-600 shadow-inner"></div>
            </div>
            <div className="text-sm font-semibold text-slate-600 cursor-default flex-1 text-center pr-20">
              Interactive AI Curriculum
            </div>
          </div>

          <div className="flex flex-1 overflow-hidden">
            
            <aside className="w-[300px] bg-slate-50/90 backdrop-blur-xl border-r border-slate-200 flex-shrink-0 h-full flex flex-col overflow-y-auto pb-8 custom-scrollbar">
              <div className="p-6 sticky top-0 bg-slate-50/90 backdrop-blur-xl z-10 border-b border-slate-200/50 mb-4">
                <Link href="/" className="flex items-center gap-3 text-slate-800 hover:text-blue-600 transition-colors">
                  <div className="p-2 bg-blue-600 rounded-lg shadow-sm">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-lg font-bold leading-tight">AI Academy</h1>
                    <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">PhD Level Engineering</p>
                  </div>
                </Link>
              </div>
              
              <nav className="px-4 space-y-8 flex-1">
                
                {curriculum.map((stack) => {
                  const Icon = (LucideIcons as any)[stack.icon] || LucideIcons.FileText;
                  const colors = colorMap[stack.color] || colorMap.blue;
                  
                  return (
                    <div key={stack.id}>
                      <div className="flex items-center gap-2 mb-2 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                        <Icon className={`w-4 h-4 ${colors.text}`} />
                        <span>{stack.title}</span>
                      </div>
                      <div className={`pl-6 space-y-1 border-l-2 ${colors.border} ml-4 mt-1`}>
                        {stack.modules.map((mod) => (
                          <Link 
                            key={mod.id} 
                            href={`/course/${stack.id}/${mod.id}`} 
                            className={`block text-[13px] text-slate-600 ${colors.hoverText} font-medium py-1.5 px-2 rounded-md ${colors.hoverBg} transition-colors`}
                            title={mod.desc}
                          >
                            {mod.title}
                          </Link>
                        ))}
                      </div>
                    </div>
                  );
                })}

              </nav>
            </aside>

            <main className="flex-1 h-full overflow-y-auto bg-white scroll-smooth custom-scrollbar relative">
              <div className="max-w-4xl mx-auto p-8 lg:p-14">
                {children}
              </div>
            </main>

          </div>
        </div>
      </body>
    </html>
  );
}
