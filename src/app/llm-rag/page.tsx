"use client";

import React from 'react';
import TerminalBlock from '@/components/TerminalBlock';

export default function LlmRag() {
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
    <div className="space-y-16 pb-24 text-slate-800">
      
      {/* Header */}
      <header className="border-b border-slate-200 pb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold uppercase tracking-widest mb-4">
          <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse"></span>
          Advanced Certification Track
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-slate-900 mb-6">Agentic AI & RAG Engineering</h1>
        <p className="text-xl text-slate-600 leading-relaxed max-w-3xl">
          Master the architecture of modern AI. Learn to design, deploy, and scale autonomous agents and Retrieval-Augmented Generation systems for production enterprise environments.
        </p>
      </header>

      {/* Module 1: Foundations */}
      <section id="foundations" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-xl">1</div>
          <h2 className="text-3xl font-bold text-slate-900">Foundations of LLMs & Attention</h2>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
            <h3 className="font-bold text-xl text-slate-800 mb-4">The Transformer Architecture</h3>
            <p className="text-sm text-slate-600 mb-4">
              The bedrock of modern AI. Unlike RNNs that read sequentially, Transformers use <strong>Self-Attention</strong> to look at the entire sequence simultaneously.
            </p>
            <div className="font-mono bg-slate-900 text-indigo-400 p-4 rounded-lg text-xs mb-4 shadow-inner">
              Attention(Q, K, V) = softmax( (Q·K^T) / √d_k ) · V
            </div>
            <p className="text-xs text-slate-500">
              Queries (Q), Keys (K), and Values (V) dynamically determine how much "focus" each word should put on every other word in the context window.
            </p>
          </div>

          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 flex flex-col justify-center">
            <h3 className="font-bold text-xl text-slate-800 mb-6">Core Concepts</h3>
            <ul className="text-sm text-slate-600 space-y-6">
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded bg-indigo-200 text-indigo-800 flex items-center justify-center font-bold shrink-0">A</div>
                <span><strong>Tokenization:</strong> Converting text into sub-word integers. (e.g., BPE, TikToken).</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded bg-indigo-200 text-indigo-800 flex items-center justify-center font-bold shrink-0">B</div>
                <span><strong>Embeddings:</strong> High-dimensional vector space mapping where semantic similarity = geometric proximity (Cosine Similarity).</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded bg-indigo-200 text-indigo-800 flex items-center justify-center font-bold shrink-0">C</div>
                <span><strong>Fine-Tuning (PEFT/LoRA):</strong> Adapting massive base models to specific tasks efficiently by freezing most weights.</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <hr className="border-slate-200" />

      {/* Module 2: Advanced RAG */}
      <section id="rag" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-xl">2</div>
          <h2 className="text-3xl font-bold text-slate-900">Retrieval-Augmented Generation (RAG)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            LLMs hallucinate and lack private data. RAG bridges this by searching a Vector Database for context <em>before</em> generating an answer.
          </p>
        </div>

        <h3 className="text-2xl font-bold mb-4 mt-8">Advanced Retrieval Techniques</h3>
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 mb-3 text-lg">Semantic Chunking</h4>
            <p className="text-sm text-slate-600">Instead of splitting by character count, split by document structure (headers, paragraphs) to preserve natural context.</p>
          </div>
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 mb-3 text-lg">Hybrid Search</h4>
            <p className="text-sm text-slate-600">Combine dense vector search (semantic meaning) with sparse keyword search (BM25) to catch exact product IDs or names.</p>
          </div>
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
            <h4 className="font-bold text-slate-900 mb-3 text-lg">Re-Ranking (Cohere)</h4>
            <p className="text-sm text-slate-600">Retrieve 50 docs cheaply, then use a Cross-Encoder to accurately score and re-rank the top 5 before passing to the LLM.</p>
          </div>
        </div>

        <h3 className="text-2xl font-bold mb-4">Python Implementation (LangChain RAG)</h3>
        <TerminalBlock language="python" filename="standard_rag_pipeline.py" code={ragCode} />
      </section>

      <hr className="border-slate-200" />

      {/* Module 3: Agents */}
      <section id="agents" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-xl">3</div>
          <h2 className="text-3xl font-bold text-slate-900">Agentic Workflows & Tool Calling</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            Moving beyond passive chatbots. Agents use frameworks like <strong>ReAct (Reason + Act)</strong> to think about a problem, decide which tools to use, execute them, and observe the results.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="p-8 border border-rose-200 bg-rose-50 rounded-2xl">
            <h4 className="font-bold text-rose-900 text-2xl mb-4">The ReAct Loop</h4>
            <ol className="text-sm text-rose-800 space-y-3 list-decimal pl-5 font-mono bg-white p-4 rounded-xl border border-rose-100 shadow-sm">
              <li><strong>Thought:</strong> I need to find the weather in NYC.</li>
              <li><strong>Action:</strong> Call \`weather_api(location="NYC")\`</li>
              <li><strong>Observation:</strong> "72F and sunny"</li>
              <li><strong>Thought:</strong> I have the answer.</li>
              <li><strong>Final Answer:</strong> It is 72F in NYC.</li>
            </ol>
          </div>

          <div className="p-8 border border-sky-200 bg-sky-50 rounded-2xl">
            <h4 className="font-bold text-sky-900 text-2xl mb-4">Function Calling</h4>
            <p className="text-sky-800 mb-4 text-base">
              Modern models (GPT-4, Claude 3.5) are natively fine-tuned to output JSON matching a strict JSON Schema you provide. 
            </p>
            <p className="text-sky-800 text-base">
              Instead of parsing regex from raw text, you give the LLM an array of available tools, and it responds with exactly which function to execute and with what arguments.
            </p>
          </div>
        </div>
      </section>

      <hr className="border-slate-200" />

      {/* Module 4: LangGraph */}
      <section id="langgraph" className="scroll-mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-xl">4</div>
          <h2 className="text-3xl font-bold text-slate-900">Multi-Agent Systems (LangGraph)</h2>
        </div>
        <div className="prose prose-slate max-w-none text-lg text-slate-600 mb-8">
          <p>
            Standard chains (LangChain) are linear (A → B → C). Real-world tasks are <strong>cyclic</strong> (Draft → Review → Revise). LangGraph treats LLM workflows as Stateful Graphs.
          </p>
        </div>

        <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 mb-8">
          <h3 className="font-bold text-slate-800 mb-4 text-xl">Why Stateful Graphs?</h3>
          <ul className="text-base text-slate-700 space-y-3 list-disc pl-5">
            <li><strong>Human-in-the-Loop:</strong> Pause execution at a node, ask a human for approval, and resume.</li>
            <li><strong>Time Travel:</strong> Because state is versioned, you can rewind a bad LLM decision to a previous state and fork the execution.</li>
            <li><strong>Specialization:</strong> Create a "Coder" agent and a "Tester" agent that bounce code back and forth until tests pass.</li>
          </ul>
        </div>

        <h3 className="text-2xl font-bold mb-4">Python Implementation (LangGraph Loop)</h3>
        <TerminalBlock language="python" filename="draft_review_loop.py" code={langGraphCode} />
      </section>

    </div>
  );
}
