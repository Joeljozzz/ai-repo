import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { BookOpen, Brain, Network, Search, CheckSquare, MessageSquare, ChevronRight } from "lucide-react";
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
    <html lang="en" className="scroll-smooth">
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
            <div className="text-sm font-semibold text-slate-600 cursor-default flex-1 text-center pr-20">
              AI Engineer's Handbook
            </div>
          </div>

          {/* Window Content */}
          <div className="flex flex-1 overflow-hidden">
            
            {/* macOS Sidebar (Translucent / Frosted Glass effect) */}
            <aside className="w-72 bg-slate-50/90 backdrop-blur-xl border-r border-slate-200 flex-shrink-0 h-full flex flex-col overflow-y-auto pb-8 custom-scrollbar">
              <div className="p-6 sticky top-0 bg-slate-50/90 backdrop-blur-xl z-10 border-b border-slate-200/50 mb-4">
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
              
              <nav className="px-4 space-y-6 flex-1">
                
                {/* Section 1 */}
                <div>
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Core Foundations</div>
                  
                  <div className="space-y-1">
                    <Link href="/ml" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-800 font-medium transition-colors">
                      <BookOpen className="w-4 h-4 text-blue-500" />
                      <span className="text-sm">Classical ML</span>
                    </Link>
                    {/* Nested Links */}
                    <div className="pl-10 space-y-1 border-l-2 border-slate-200 ml-4 mt-1">
                      <Link href="/ml#linear" className="block text-xs text-slate-500 hover:text-blue-600 py-1">Linear & Distance Models</Link>
                      <Link href="/ml#svm" className="block text-xs text-slate-500 hover:text-blue-600 py-1">Support Vector Machines</Link>
                      <Link href="/ml#trees" className="block text-xs text-slate-500 hover:text-blue-600 py-1">Decision Trees</Link>
                      <Link href="/ml#ensembles" className="block text-xs text-slate-500 hover:text-blue-600 py-1">Bagging & Boosting</Link>
                    </div>
                  </div>

                  <div className="space-y-1 mt-2">
                    <Link href="/evaluations" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-800 font-medium transition-colors">
                      <CheckSquare className="w-4 h-4 text-emerald-500" />
                      <span className="text-sm">Evaluations</span>
                    </Link>
                  </div>
                </div>

                {/* Section 2 */}
                <div>
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Advanced Models</div>
                  
                  <div className="space-y-1">
                    <Link href="/deep-learning" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-800 font-medium transition-colors">
                      <Network className="w-4 h-4 text-purple-500" />
                      <span className="text-sm">Deep Learning</span>
                    </Link>
                    <div className="pl-10 space-y-1 border-l-2 border-slate-200 ml-4 mt-1">
                      <Link href="/deep-learning#mlp" className="block text-xs text-slate-500 hover:text-purple-600 py-1">Theory & Math (MLP)</Link>
                      <Link href="/deep-learning#cnn" className="block text-xs text-slate-500 hover:text-purple-600 py-1">CNNs (Vision)</Link>
                      <Link href="/deep-learning#rnn" className="block text-xs text-slate-500 hover:text-purple-600 py-1">RNNs & LSTMs</Link>
                    </div>
                  </div>

                  <div className="space-y-1 mt-2">
                    <Link href="/search-models" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-800 font-medium transition-colors">
                      <Search className="w-4 h-4 text-amber-500" />
                      <span className="text-sm">Search & Game Theory</span>
                    </Link>
                  </div>
                </div>

                {/* Section 3 */}
                <div>
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Generative AI</div>
                  
                  <div className="space-y-1">
                    <Link href="/llm-rag" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-slate-200/50 text-slate-800 font-medium transition-colors">
                      <MessageSquare className="w-4 h-4 text-indigo-500" />
                      <span className="text-sm">LLMs & RAG</span>
                    </Link>
                    <div className="pl-10 space-y-1 border-l-2 border-slate-200 ml-4 mt-1">
                      <Link href="/llm-rag#foundations" className="block text-xs text-slate-500 hover:text-indigo-600 py-1">Transformers & LLMs</Link>
                      <Link href="/llm-rag#rag" className="block text-xs text-slate-500 hover:text-indigo-600 py-1">Advanced RAG</Link>
                      <Link href="/llm-rag#agents" className="block text-xs text-slate-500 hover:text-indigo-600 py-1">Agentic Workflows</Link>
                      <Link href="/llm-rag#langgraph" className="block text-xs text-slate-500 hover:text-indigo-600 py-1">LangGraph & Multi-Agent</Link>
                    </div>
                  </div>
                </div>

              </nav>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 h-full overflow-y-auto bg-white scroll-smooth custom-scrollbar">
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
