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
  description: "Stanford-level AI curriculum from statistics to production AI systems.",
};

const colorMap: Record<string, { dot: string; text: string; border: string; hover: string; hoverBg: string }> = {
  sky:    { dot: "bg-sky-400",    text: "text-sky-600",    border: "border-sky-200",    hover: "hover:text-sky-700",    hoverBg: "hover:bg-sky-50" },
  emerald:{ dot: "bg-emerald-400",text: "text-emerald-600",border: "border-emerald-200",hover: "hover:text-emerald-700",hoverBg: "hover:bg-emerald-50" },
  purple: { dot: "bg-purple-400", text: "text-purple-600", border: "border-purple-200", hover: "hover:text-purple-700", hoverBg: "hover:bg-purple-50" },
  indigo: { dot: "bg-indigo-400", text: "text-indigo-600", border: "border-indigo-200", hover: "hover:text-indigo-700", hoverBg: "hover:bg-indigo-50" },
  teal:   { dot: "bg-teal-400",   text: "text-teal-600",   border: "border-teal-200",   hover: "hover:text-teal-700",   hoverBg: "hover:bg-teal-50" },
  rose:   { dot: "bg-rose-400",   text: "text-rose-600",   border: "border-rose-200",   hover: "hover:text-rose-700",   hoverBg: "hover:bg-rose-50" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} bg-gradient-to-br from-slate-200 via-blue-100 to-indigo-200 flex h-screen w-screen items-center justify-center p-3 sm:p-6 overflow-hidden`}>
        
        <div className="w-full h-full max-w-[1400px] flex flex-col bg-[#fafafa] rounded-2xl shadow-2xl overflow-hidden ring-1 ring-slate-900/10">
          
          {/* macOS Title Bar */}
          <div className="h-11 bg-white border-b border-slate-200 flex items-center px-4 shrink-0 select-none">
            <div className="flex space-x-2 w-24">
              <div className="w-3 h-3 rounded-full bg-[#ff5f57]"></div>
              <div className="w-3 h-3 rounded-full bg-[#febc2e]"></div>
              <div className="w-3 h-3 rounded-full bg-[#28c840]"></div>
            </div>
            <div className="text-xs font-semibold text-slate-400 flex-1 text-center tracking-wide uppercase">
              AI Engineer&apos;s Handbook — Full Curriculum
            </div>
          </div>

          <div className="flex flex-1 overflow-hidden">
            
            {/* Sidebar */}
            <aside className="w-72 bg-white border-r border-slate-100 flex-shrink-0 flex flex-col overflow-y-auto custom-scrollbar">
              <div className="px-5 py-4 border-b border-slate-100">
                <Link href="/" className="flex items-center gap-3 group">
                  <div className="p-1.5 bg-slate-900 rounded-lg">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-900 group-hover:text-blue-600 transition-colors">AI Engineer&apos;s Handbook</div>
                    <div className="text-[10px] text-slate-400 uppercase tracking-widest font-medium">Zero → Production AI</div>
                  </div>
                </Link>
              </div>
              
              <nav className="py-4 px-3 space-y-6 flex-1">
                {curriculum.map((stack) => {
                  const Icon = (LucideIcons as any)[stack.icon] || LucideIcons.BookOpen;
                  const c = colorMap[stack.color] || colorMap.sky;
                  return (
                    <div key={stack.id}>
                      <div className="flex items-center gap-2 px-2 mb-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`}></span>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{stack.label} · {stack.title}</span>
                      </div>
                      <div className={`ml-3 pl-4 border-l ${c.border} space-y-0.5`}>
                        {stack.modules.map((mod) => (
                          <Link
                            key={mod.id}
                            href={`/course/${stack.id}/${mod.id}`}
                            className={`block text-[13px] text-slate-500 ${c.hover} ${c.hoverBg} font-medium py-1.5 px-2 rounded-md transition-colors`}
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

            {/* Content */}
            <main className="flex-1 h-full overflow-y-auto bg-[#fafafa] custom-scrollbar">
              <div className="max-w-3xl mx-auto px-8 py-12 lg:px-16">
                {children}
              </div>
            </main>

          </div>
        </div>
      </body>
    </html>
  );
}
