import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Learning Repository",
  description: "A comprehensive guide to ML, Deep Learning, and AI Architectures",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 text-gray-900 flex min-h-screen`}>
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r border-gray-200 flex-shrink-0 hidden md:block">
          <div className="p-6">
            <Link href="/" className="text-xl font-bold text-blue-600">AI Repo</Link>
            <p className="text-sm text-gray-500 mt-1">Concepts & Architectures</p>
          </div>
          <nav className="px-4 py-2 space-y-1">
            <Link href="/ml" className="block px-3 py-2 rounded-md hover:bg-gray-100 font-medium">Classical ML</Link>
            <Link href="/deep-learning" className="block px-3 py-2 rounded-md hover:bg-gray-100 font-medium">Deep Learning</Link>
            <Link href="/search-models" className="block px-3 py-2 rounded-md hover:bg-gray-100 font-medium">Search Models</Link>
            <Link href="/evaluations" className="block px-3 py-2 rounded-md hover:bg-gray-100 font-medium">Evaluations</Link>
            <Link href="/llm-rag" className="block px-3 py-2 rounded-md hover:bg-gray-100 font-medium">LLMs & RAG</Link>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-auto">
          <div className="max-w-4xl mx-auto bg-white shadow-sm rounded-xl p-8 border border-gray-100">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
