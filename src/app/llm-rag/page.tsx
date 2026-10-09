"use client";

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function LlmRag() {
  const [activeModule, setActiveModule] = useState('mod1');

  const ragCode = `from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 1. Load and Chunk
loader = PyPDFLoader("enterprise_data.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 2. Embed and Index
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 3. Formulate RAG Chain
llm = ChatOpenAI(model="gpt-4-turbo")
template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 4. Invoke
response = rag_chain.invoke("What is our Q3 revenue strategy?")
print(response.content)`;

  const langGraphCode = `from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# 1. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    needs_revision: bool

# 2. Define Nodes (The "Actors")
def drafter(state: AgentState):
    """Generates the initial response."""
    last_msg = state['messages'][-1].content
    # (LLM Call omitted for brevity)
    draft = f"Drafted response for: {last_msg}"
    return {"messages": [HumanMessage(content=draft)]}

def reviewer(state: AgentState):
    """Critiques the draft and flags if revision is needed."""
    # (LLM Call to check for hallucinations/style)
    needs_rev = True # Mocking a failure
    return {"needs_revision": needs_rev}

# 3. Define Conditional Edges
def should_continue(state: AgentState):
    if state["needs_revision"]:
        return "drafter" # Loop back!
    return END

# 4. Build the Cyclic Graph
workflow = StateGraph(AgentState)
workflow.add_node("drafter", drafter)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("drafter")
workflow.add_edge("drafter", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

# 5. Compile and Run
app = workflow.compile()
final_state = app.invoke({"messages": [HumanMessage(content="Write a marketing email.")]})`;

  return (
    <div className="space-y-12 pb-16 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse"></span>
          Advanced Certification Track
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-4">Agentic AI & RAG Engineering</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Master the architecture of modern AI. Learn to design, deploy, and scale autonomous agents and Retrieval-Augmented Generation systems for production enterprise environments.
        </p>
      </header>

      {/* Course Modules Navigation */}
      <div className="flex border-b border-slate-200 gap-8 overflow-x-auto no-scrollbar">
        {[
          { id: 'mod1', title: 'Module 1', subtitle: 'LLM Foundations' },
          { id: 'mod2', title: 'Module 2', subtitle: 'Advanced RAG' },
          { id: 'mod3', title: 'Module 3', subtitle: 'Agentic Workflows' },
          { id: 'mod4', title: 'Module 4', subtitle: 'Multi-Agent (LangGraph)' },
        ].map((mod) => (
          <button 
            key={mod.id}
            onClick={() => setActiveModule(mod.id)}
            className={`pb-4 text-left transition-colors whitespace-nowrap min-w-[150px] ${
              activeModule === mod.id 
                ? 'border-b-2 border-indigo-600' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <div className={`text-xs font-bold uppercase tracking-wider mb-1 ${activeModule === mod.id ? 'text-indigo-600' : 'text-slate-400'}`}>
              {mod.title}
            </div>
            <div className={`text-base font-medium ${activeModule === mod.id ? 'text-slate-900' : ''}`}>
              {mod.subtitle}
            </div>
          </button>
        ))}
      </div>

      {/* Module 1: Foundations */}
      {activeModule === 'mod1' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-6 text-slate-900">Foundations of LLMs & Attention</h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                <h3 className="font-bold text-xl text-slate-800 mb-2">The Transformer Architecture</h3>
                <p className="text-sm text-slate-600 mb-4">
                  The bedrock of modern AI. Unlike RNNs that read sequentially, Transformers use <strong>Self-Attention</strong> to look at the entire sequence simultaneously.
                </p>
                <div className="font-mono bg-slate-900 text-indigo-400 p-4 rounded-lg text-xs mb-4">
                  Attention(Q, K, V) = softmax( (Q·K^T) / √d_k ) · V
                </div>
                <p className="text-xs text-slate-500">
                  Queries (Q), Keys (K), and Values (V) dynamically determine how much "focus" each word should put on every other word in the context window.
                </p>
              </div>

              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col justify-center">
                <h3 className="font-bold text-xl text-slate-800 mb-4">Core Concepts</h3>
                <ul className="text-sm text-slate-600 space-y-4">
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">1.</span>
                    <span><strong>Tokenization:</strong> Converting text into sub-word integers. (e.g., BPE, TikToken).</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">2.</span>
                    <span><strong>Embeddings:</strong> High-dimensional vector space mapping where semantic similarity = geometric proximity (Cosine Similarity).</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">3.</span>
                    <span><strong>Fine-Tuning (PEFT/LoRA):</strong> Adapting massive base models to specific tasks efficiently by freezing most weights.</span>
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 2: Advanced RAG */}
      {activeModule === 'mod2' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Retrieval-Augmented Generation (RAG)</h2>
            <p className="text-lg text-slate-600 mb-8">
              LLMs hallucinate and lack private data. RAG bridges this by searching a Vector Database for context <em>before</em> generating an answer.
            </p>

            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <div className="bg-slate-900 text-slate-400 text-xs px-4 py-2 font-mono border-b border-slate-800">
                standard_rag_pipeline.py
              </div>
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem', backgroundColor: '#0f172a'}}>
                {ragCode}
              </SyntaxHighlighter>
            </div>

            <h3 className="text-xl font-bold mb-4 mt-12">Advanced Retrieval Techniques</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">Semantic Chunking</h4>
                <p className="text-sm text-slate-600">Instead of splitting by character count, split by document structure (headers, paragraphs) to preserve natural context.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">Hybrid Search</h4>
                <p className="text-sm text-slate-600">Combine dense vector search (semantic meaning) with sparse keyword search (BM25) to catch exact product IDs or names.</p>
              </div>
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-2">Re-Ranking (Cohere)</h4>
                <p className="text-sm text-slate-600">Retrieve 50 docs cheaply, then use a Cross-Encoder to accurately score and re-rank the top 5 before passing to the LLM.</p>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 3: Agentic Workflows */}
      {activeModule === 'mod3' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Agentic Workflows & Tool Calling</h2>
            <p className="text-lg text-slate-600 mb-6">
              Moving beyond passive chatbots. Agents use frameworks like <strong>ReAct (Reason + Act)</strong> to think about a problem, decide which tools to use, execute them, and observe the results.
            </p>

            <div className="grid md:grid-cols-2 gap-8 mb-6">
              <div className="p-5 border-l-4 border-rose-500 bg-rose-50/50 rounded-r-lg">
                <h4 className="font-bold text-rose-900 text-lg mb-2">The ReAct Loop</h4>
                <ol className="text-sm text-rose-800 space-y-2 list-decimal pl-5 font-mono">
                  <li><strong>Thought:</strong> I need to find the weather in NYC.</li>
                  <li><strong>Action:</strong> Call \`weather_api(location="NYC")\`</li>
                  <li><strong>Observation:</strong> "72F and sunny"</li>
                  <li><strong>Thought:</strong> I have the answer.</li>
                  <li><strong>Final Answer:</strong> It is 72F in NYC.</li>
                </ol>
              </div>

              <div className="p-5 border-l-4 border-sky-500 bg-sky-50/50 rounded-r-lg">
                <h4 className="font-bold text-sky-900 text-lg mb-2">Function / Tool Calling</h4>
                <p className="text-sky-800 mb-2 text-sm">
                  Modern models (GPT-4, Claude 3.5) are natively fine-tuned to output JSON matching a strict JSON Schema you provide. 
                </p>
                <p className="text-sky-800 text-sm">
                  Instead of parsing regex from raw text, you give the LLM an array of available tools, and it responds with exactly which function to execute and with what arguments.
                </p>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* Module 4: Multi-Agent */}
      {activeModule === 'mod4' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <section className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-3xl font-bold mb-4 text-slate-900">Multi-Agent Systems (LangGraph)</h2>
            <p className="text-lg text-slate-600 mb-8">
              Standard chains (LangChain) are linear (A → B → C). Real-world tasks are <strong>cyclic</strong> (Draft → Review → Revise). LangGraph treats LLM workflows as Stateful Graphs.
            </p>

            <h3 className="text-xl font-bold mb-4">Building a Cyclic Agent in LangGraph</h3>
            <div className="rounded-xl overflow-hidden shadow-lg border border-slate-800 mb-8">
              <div className="bg-slate-900 text-slate-400 text-xs px-4 py-2 font-mono border-b border-slate-800">
                draft_review_loop.py
              </div>
              <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin: 0, padding: '1.5rem', backgroundColor: '#0f172a'}}>
                {langGraphCode}
              </SyntaxHighlighter>
            </div>

            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
              <h3 className="font-bold text-slate-800 mb-3">Why Stateful Graphs?</h3>
              <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
                <li><strong>Human-in-the-Loop:</strong> Pause execution at a node, ask a human for approval, and resume.</li>
                <li><strong>Time Travel:</strong> Because state is versioned, you can rewind a bad LLM decision to a previous state and fork the execution.</li>
                <li><strong>Specialization:</strong> Create a "Coder" agent and a "Tester" agent that bounce code back and forth until tests pass.</li>
              </ul>
            </div>
          </section>
        </div>
      )}

    </div>
  );
}
