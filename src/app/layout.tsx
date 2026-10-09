import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { BookOpen, Brain, Network, Search, CheckSquare, MessageSquare } from "lucide-react";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Engineer's Handbook",
  description: "Comprehensive guide to ML, Deep Learning, and AI Architectures",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* 
        The body represents the "Mac Desktop" background. 
        Using a nice, subtle macOS-like blurred gradient wallpaper effect.
      */}
      <body className={`${inter.className} bg-gradient-to-br from-indigo-300 via-purple-300 to-pink-300 text-slate-900 flex h-screen w-screen items-center justify-center p-4 sm:p-8 overflow-hidden`}>
        
        {/* The macOS Window Container */}
        <div className="w-full h-full max-w-7xl max-h-[900px] flex flex-col bg-white rounded-xl shadow-2xl overflow-hidden ring-1 ring-slate-900/10">
          
          {/* macOS Title Bar */}
          <div className="h-12 bg-slate-100 border-b border-slate-200 flex items-center px-4 shrink-0 justify-between select-none">
            {/* Traffic Lights */}
            <div className="flex space-x-2 w-20">
              <div className="w-3.5 h-3.5 rounded-full bg-red-500 border border-red-600 shadow-inner"></div>
              <div className="w-3.5 h-3.5 rounded-full bg-amber-500 border border-amber-600 shadow-inner"></div>
              <div className="w-3.5 h-3.5 rounded-full bg-emerald-500 border border-emerald-600 shadow-inner"></div>
            </div>
            {/* Window Title */}
            <div className="text-sm font-semibold text-slate-600 cursor-default flex-1 text-center pr-20">
              AI Engineer's Handbook
            </div>
          </div>

          {/* Window Content */}
          <div className="flex flex-1 overflow-hidden">
            
            {/* macOS Sidebar (Translucent / Frosted Glass effect) */}
            <aside className="w-64 bg-slate-50/80 backdrop-blur-xl border-r border-slate-200 flex-shrink-0 h-full flex flex-col overflow-y-auto">
              <div className="p-6">
                <Link href="/" className="flex items-center gap-3 text-slate-800 hover:text-blue-600 transition-colors">
                  <div className="p-2 bg-blue-600 rounded-lg shadow-sm">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-lg font-bold leading-tight">AI Handbook</h1>
                    <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">Zero to AI Engineer</p>
                  </div>
                </Link>
              </div>
              
              <nav className="px-3 space-y-1 flex-1">
                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 mt-2 px-3">Core Foundations</div>
                
                <Link href="/ml" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-700 transition-colors">
                  <BookOpen className="w-4 h-4 text-blue-500" />
                  <span className="font-medium text-sm">Classical ML</span>
                </Link>
                
                <Link href="/evaluations" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-700 transition-colors">
                  <CheckSquare className="w-4 h-4 text-emerald-500" />
                  <span className="font-medium text-sm">Evaluations</span>
                </Link>

                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 mt-6 px-3">Advanced Models</div>
                
                <Link href="/deep-learning" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-700 transition-colors">
                  <Network className="w-4 h-4 text-purple-500" />
                  <span className="font-medium text-sm">Deep Learning</span>
                </Link>
                
                <Link href="/search-models" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-700 transition-colors">
                  <Search className="w-4 h-4 text-amber-500" />
                  <span className="font-medium text-sm">Game Theory</span>
                </Link>
                
                <Link href="/llm-rag" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-700 transition-colors">
                  <MessageSquare className="w-4 h-4 text-indigo-500" />
                  <span className="font-medium text-sm">LLMs & RAG</span>
                </Link>
              </nav>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 h-full overflow-y-auto bg-white">
              <div className="max-w-4xl mx-auto p-8 lg:p-12">
                {children}
              </div>
            </main>

          </div>
        </div>
      </body>
    </html>
  );
}
