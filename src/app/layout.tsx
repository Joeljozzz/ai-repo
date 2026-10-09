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
      <body className={`${inter.className} bg-slate-50 text-slate-900 flex h-screen overflow-hidden`}>
        {/* Sidebar */}
        <aside className="w-72 bg-slate-900 text-slate-300 flex-shrink-0 h-full flex flex-col overflow-y-auto">
          <div className="p-6 border-b border-slate-800">
            <Link href="/" className="flex items-center gap-3 text-white hover:text-blue-400 transition-colors">
              <Brain className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-xl font-bold">AI Handbook</h1>
                <p className="text-xs text-slate-400">Zero to AI Engineer</p>
              </div>
            </Link>
          </div>
          <nav className="p-4 space-y-2 flex-1">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 mt-4">Core Foundations</div>
            
            <Link href="/ml" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 hover:text-white transition-colors">
              <BookOpen className="w-5 h-5" />
              <span className="font-medium">Classical ML</span>
            </Link>
            
            <Link href="/evaluations" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 hover:text-white transition-colors">
              <CheckSquare className="w-5 h-5" />
              <span className="font-medium">Evaluations & Metrics</span>
            </Link>

            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 mt-8">Advanced Models</div>
            
            <Link href="/deep-learning" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 hover:text-white transition-colors">
              <Network className="w-5 h-5" />
              <span className="font-medium">Deep Learning</span>
            </Link>
            
            <Link href="/search-models" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 hover:text-white transition-colors">
              <Search className="w-5 h-5" />
              <span className="font-medium">Search & Game Theory</span>
            </Link>
            
            <Link href="/llm-rag" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 hover:text-white transition-colors">
              <MessageSquare className="w-5 h-5" />
              <span className="font-medium">LLMs, LangChain & RAG</span>
            </Link>
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 h-full overflow-y-auto bg-slate-50">
          <div className="max-w-5xl mx-auto p-8 lg:p-12">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
