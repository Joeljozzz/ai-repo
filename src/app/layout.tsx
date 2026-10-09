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
              Interactive AI Curriculum
            </div>
          </div>

          {/* Window Content */}
          <div className="flex flex-1 overflow-hidden">
            
            {/* macOS Sidebar */}
            <aside className="w-72 bg-slate-50/90 backdrop-blur-xl border-r border-slate-200 flex-shrink-0 h-full flex flex-col overflow-y-auto pb-8 custom-scrollbar">
              <div className="p-6 sticky top-0 bg-slate-50/90 backdrop-blur-xl z-10 border-b border-slate-200/50 mb-4">
                <Link href="/" className="flex items-center gap-3 text-slate-800 hover:text-blue-600 transition-colors">
                  <div className="p-2 bg-blue-600 rounded-lg shadow-sm">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-lg font-bold leading-tight">AI Academy</h1>
                    <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">Zero to AI Engineer</p>
                  </div>
                </Link>
              </div>
              
              <nav className="px-4 space-y-6 flex-1">
                
                {/* Stack 1: Classical ML */}
                <div>
                  <div className="flex items-center gap-2 mb-2 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    <BookOpen className="w-4 h-4 text-emerald-500" />
                    <span>Stack 1: Classical ML</span>
                  </div>
                  <div className="pl-6 space-y-1 border-l-2 border-emerald-100 ml-4 mt-1">
                    <Link href="/course/ml/linear-models" className="block text-[13px] text-slate-600 hover:text-emerald-600 font-medium py-1.5 px-2 rounded-md hover:bg-emerald-50 transition-colors">1.1 Linear & Distance Models</Link>
                    <Link href="/course/ml/svm" className="block text-[13px] text-slate-600 hover:text-emerald-600 font-medium py-1.5 px-2 rounded-md hover:bg-emerald-50 transition-colors">1.2 Support Vector Machines</Link>
                    <Link href="/course/ml/trees" className="block text-[13px] text-slate-600 hover:text-emerald-600 font-medium py-1.5 px-2 rounded-md hover:bg-emerald-50 transition-colors">1.3 Decision Trees</Link>
                    <Link href="/course/ml/ensembles" className="block text-[13px] text-slate-600 hover:text-emerald-600 font-medium py-1.5 px-2 rounded-md hover:bg-emerald-50 transition-colors">1.4 Bagging & Boosting</Link>
                  </div>
                </div>

                {/* Stack 2: Deep Learning */}
                <div>
                  <div className="flex items-center gap-2 mb-2 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    <Network className="w-4 h-4 text-purple-500" />
                    <span>Stack 2: Deep Learning</span>
                  </div>
                  <div className="pl-6 space-y-1 border-l-2 border-purple-100 ml-4 mt-1">
                    <Link href="/course/dl/foundations" className="block text-[13px] text-slate-600 hover:text-purple-600 font-medium py-1.5 px-2 rounded-md hover:bg-purple-50 transition-colors">2.1 MLP Foundations</Link>
                    <Link href="/course/dl/cnn" className="block text-[13px] text-slate-600 hover:text-purple-600 font-medium py-1.5 px-2 rounded-md hover:bg-purple-50 transition-colors">2.2 Computer Vision (CNN)</Link>
                    <Link href="/course/dl/rnn" className="block text-[13px] text-slate-600 hover:text-purple-600 font-medium py-1.5 px-2 rounded-md hover:bg-purple-50 transition-colors">2.3 Sequence Modeling</Link>
                    <Link href="/course/dl/attention" className="block text-[13px] text-slate-600 hover:text-purple-600 font-medium py-1.5 px-2 rounded-md hover:bg-purple-50 transition-colors">2.4 Attention Mechanisms</Link>
                  </div>
                </div>

                {/* Stack 3: LLMs */}
                <div>
                  <div className="flex items-center gap-2 mb-2 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    <MessageSquare className="w-4 h-4 text-indigo-500" />
                    <span>Stack 3: LLMs & GenAI</span>
                  </div>
                  <div className="pl-6 space-y-1 border-l-2 border-indigo-100 ml-4 mt-1">
                    <Link href="/course/llm/transformers" className="block text-[13px] text-slate-600 hover:text-indigo-600 font-medium py-1.5 px-2 rounded-md hover:bg-indigo-50 transition-colors">3.1 Transformers Architecture</Link>
                    <Link href="/course/llm/fine-tuning" className="block text-[13px] text-slate-600 hover:text-indigo-600 font-medium py-1.5 px-2 rounded-md hover:bg-indigo-50 transition-colors">3.2 LoRA & Fine-Tuning</Link>
                    <Link href="/course/llm/rag" className="block text-[13px] text-slate-600 hover:text-indigo-600 font-medium py-1.5 px-2 rounded-md hover:bg-indigo-50 transition-colors">3.3 Advanced RAG</Link>
                    <Link href="/course/llm/agents" className="block text-[13px] text-slate-600 hover:text-indigo-600 font-medium py-1.5 px-2 rounded-md hover:bg-indigo-50 transition-colors">3.4 Agentic Frameworks</Link>
                    <Link href="/course/llm/multi-agent" className="block text-[13px] text-slate-600 hover:text-indigo-600 font-medium py-1.5 px-2 rounded-md hover:bg-indigo-50 transition-colors">3.5 LangGraph Multi-Agent</Link>
                  </div>
                </div>

                {/* Stack 4: MLOps */}
                <div>
                  <div className="flex items-center gap-2 mb-2 px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    <CheckSquare className="w-4 h-4 text-rose-500" />
                    <span>Stack 4: MLOps & Evals</span>
                  </div>
                  <div className="pl-6 space-y-1 border-l-2 border-rose-100 ml-4 mt-1">
                    <Link href="/course/ops/metrics" className="block text-[13px] text-slate-600 hover:text-rose-600 font-medium py-1.5 px-2 rounded-md hover:bg-rose-50 transition-colors">4.1 Classification Metrics</Link>
                    <Link href="/course/ops/llm-evals" className="block text-[13px] text-slate-600 hover:text-rose-600 font-medium py-1.5 px-2 rounded-md hover:bg-rose-50 transition-colors">4.2 LLM Evaluations</Link>
                    <Link href="/course/ops/deployment" className="block text-[13px] text-slate-600 hover:text-rose-600 font-medium py-1.5 px-2 rounded-md hover:bg-rose-50 transition-colors">4.3 Serving & Scaling</Link>
                  </div>
                </div>

              </nav>
            </aside>

            {/* Main Content Area */}
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
