import os
import json
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from duckduckgo_search import DDGS
from openai import OpenAI

app = FastAPI(title="Agentic RAG Engine")

# ---------------------------------------------------------------------------
# Resilient In-Memory Vector & Semantic Knowledge Base (Pure Python)
# ---------------------------------------------------------------------------
class PurePythonVectorDB:
    def __init__(self):
        self.docs = []

    def add(self, doc_id: str, title: str, content: str):
        self.docs = [d for d in self.docs if d["id"] != doc_id]
        self.docs.append({"id": doc_id, "title": title, "content": content})

    def count(self):
        return len(self.docs)

    def get_all(self):
        return self.docs

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def query(self, query_text: str, n_results: int = 2) -> List[Dict[str, str]]:
        q_tokens = self._tokenize(query_text)
        if not q_tokens or not self.docs:
            return self.docs[:n_results]

        q_counter = Counter(q_tokens)
        scored = []
        for doc in self.docs:
            d_tokens = self._tokenize(doc["title"] + " " + doc["content"])
            d_counter = Counter(d_tokens)

            # Cosine similarity over term frequencies
            intersection = set(q_counter.keys()) & set(d_counter.keys())
            dot = sum(q_counter[t] * d_counter[t] for t in intersection)
            norm_q = math.sqrt(sum(v ** 2 for v in q_counter.values()))
            norm_d = math.sqrt(sum(v ** 2 for v in d_counter.values()))
            score = (dot / (norm_q * norm_d)) if (norm_q and norm_d) else 0.0
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:n_results]]

db = PurePythonVectorDB()

SAMPLE_DOCS = [
    {
        "id": "DOC-AUTH-001",
        "title": "OAuth 2.0 & Session Security Spec (Internal v2.4)",
        "content": (
            "Our primary API uses OAuth 2.0 with JWT tokens. The internal refresh token "
            "lifetime is strictly 7 days. Bearer tokens must be passed in the Authorization header. "
            "Rate limits for internal service accounts are capped at 10,000 requests per minute."
        )
    },
    {
        "id": "DOC-BILL-002",
        "title": "Stripe Gateway Integration Policy",
        "content": (
            "We run on Stripe API version 2023-10-16. All customer subscription changes "
            "trigger webhook 'customer.subscription.updated'. We strictly enforce proration=always_invoice. "
            "Crypto payments are disabled as per compliance policy section 4."
        )
    },
    {
        "id": "DOC-INFRA-003",
        "title": "Database Connection Pooling Guidelines",
        "content": (
            "PostgreSQL instances are deployed on AWS Aurora. All Microservices must connect "
            "via PgBouncer. Direct database connections without SSL are rejected by security group SG-99201."
        )
    }
]

for doc in SAMPLE_DOCS:
    db.add(doc["id"], doc["title"], doc["content"])

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def tool_search_internal_docs(query: str, n_results: int = 2) -> List[Dict[str, Any]]:
    results = db.query(query, n_results=n_results)
    return [{
        "source_type": "INTERNAL",
        "id": d["id"],
        "title": d["title"],
        "content": d["content"]
    } for d in results]

def tool_search_web(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "source_type": "WEB",
                    "title": r.get("title", "Web Result"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        results.append({
            "source_type": "WEB",
            "title": "Search Error / Offline Mode",
            "url": "https://duckduckgo.com",
            "snippet": f"Web query unavailable or rate-limited: {str(e)}"
        })
    return results

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_internal_docs",
            "description": "Search private internal enterprise specs, company guides, and internal architecture documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search keywords or question."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live public internet for recent changes, breaking news, third-party vendor updates, or general public knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The web search query string."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": "Call this if the query is excessively vague, ambiguous, or lacks context to select the right source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_to_user": {"type": "string", "description": "Clarification question to ask."}
                },
                "required": ["question_to_user"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an Agentic RAG System equipped with tools.
You MUST decide whether to search internal company documents, the public web, both, or ask for clarification.

Rules:
1. Distinguish between internal docs and external web info.
2. If the user asks about our internal system/policy AND recent external updates (e.g. Stripe, AWS, Python), call BOTH tools.
3. Every claim supported by internal docs MUST cite: [INTERNAL: <doc_id>]
4. Every claim supported by external web search MUST cite: [WEB: <url>]
5. NEVER invent citations or document IDs. If information was not returned by a tool, state that you do not know.
"""

def audit_and_verify_citations(answer: str, context_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_internal_ids = {s["id"] for s in context_sources if s.get("source_type") == "INTERNAL"}
    valid_web_urls = {s["url"] for s in context_sources if s.get("source_type") == "WEB"}

    internal_citations_found = re.findall(r"\[INTERNAL:\s*([a-zA-Z0-9_\-]+)\]", answer)
    web_citations_found = re.findall(r"\[WEB:\s*([^\]]+)\]", answer)

    invalid_citations = []
    for cid in internal_citations_found:
        if cid not in valid_internal_ids:
            invalid_citations.append(f"[INTERNAL: {cid}]")

    for url in web_citations_found:
        if url not in valid_web_urls and not any(url in w for w in valid_web_urls):
            invalid_citations.append(f"[WEB: {url}]")

    return {
        "is_safe": len(invalid_citations) == 0,
        "invalid_citations": invalid_citations,
        "internal_citations_count": len(internal_citations_found),
        "web_citations_count": len(web_citations_found)
    }

class QueryRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    custom_model: Optional[str] = "gpt-4o-mini"

class NewDocRequest(BaseModel):
    id: str
    title: str
    content: str

@app.post("/api/documents")
def add_custom_document(doc: NewDocRequest):
    db.add(doc.id, doc.title, doc.content)
    return {"status": "success", "message": f"Document {doc.id} indexed."}

@app.get("/api/documents")
def get_documents():
    return db.get_all()

@app.post("/api/chat")
def run_agentic_rag(req: QueryRequest):
    api_key = req.api_key or os.getenv("OPENAI_API_KEY")
    reasoning_trace = []
    collected_sources = []

    if not api_key:
        return run_mock_agent(req.message)

    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.message}
    ]

    reasoning_trace.append({"step": "Plan", "detail": "Analyzing user query for autonomous routing..."})

    try:
        response = client.chat.completions.create(
            model=req.custom_model or "gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        reasoning_trace.append({"step": "Decision", "detail": "Direct response without external queries."})
        return {
            "answer": response_message.content,
            "trace": reasoning_trace,
            "sources": [],
            "audit": {"is_safe": True, "invalid_citations": [], "internal_citations_count": 0, "web_citations_count": 0}
        }

    messages.append(response_message)
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        if func_name == "request_clarification":
            clarification = func_args.get("question_to_user", "Could you provide more context?")
            reasoning_trace.append({"step": "Tool Call", "detail": f"Agent requested clarification: '{clarification}'"})
            return {
                "answer": f"**Clarification needed:** {clarification}",
                "trace": reasoning_trace,
                "sources": [],
                "audit": {"is_safe": True, "invalid_citations": [], "internal_citations_count": 0, "web_citations_count": 0}
            }

        elif func_name == "search_internal_docs":
            q = func_args.get("query", "")
            reasoning_trace.append({"step": "Tool Call: Internal Docs", "detail": f"Searching KB for: '{q}'"})
            docs = tool_search_internal_docs(q)
            collected_sources.extend(docs)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(docs)
            })

        elif func_name == "search_web":
            q = func_args.get("query", "")
            reasoning_trace.append({"step": "Tool Call: Web Search", "detail": f"Searching live web for: '{q}'"})
            web_results = tool_search_web(q)
            collected_sources.extend(web_results)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(web_results)
            })

    reasoning_trace.append({"step": "Synthesis", "detail": "Synthesizing retrieved context with strict citation enforcement."})
    final_response = client.chat.completions.create(
        model=req.custom_model or "gpt-4o-mini",
        messages=messages
    )
    final_answer = final_response.choices[0].message.content

    audit = audit_and_verify_citations(final_answer, collected_sources)
    if not audit["is_safe"]:
        reasoning_trace.append({
            "step": "Safeguard Alert",
            "detail": f"Unverified citations detected: {audit['invalid_citations']}"
        })

    return {
        "answer": final_answer,
        "trace": reasoning_trace,
        "sources": collected_sources,
        "audit": audit
    }

def run_mock_agent(message: str):
    m_lower = message.lower()
    trace = [{"step": "Routing Logic (Demo Mode)", "detail": "Evaluating keywords for source intent."}]
    sources = []

    need_internal = any(k in m_lower for k in ["internal", "our", "doc", "spec", "policy", "stripe", "oauth", "db", "aurora"])
    need_web = any(k in m_lower for k in ["recent", "latest", "api change", "news", "updated", "today"])

    if not need_internal and not need_web and len(message.split()) < 3:
        return {
            "answer": "Could you please clarify whether you are asking about our internal corporate documentation or recent public updates?",
            "trace": [{"step": "Clarification", "detail": "Prompt lacked sufficient detail."}],
            "sources": [],
            "audit": {"is_safe": True, "invalid_citations": [], "internal_citations_count": 0, "web_citations_count": 0}
        }

    answer_parts = []
    if need_internal:
        trace.append({"step": "Tool Call: Internal Docs", "detail": "Retrieved internal documentation specs."})
        docs = tool_search_internal_docs(message)
        sources.extend(docs)
        if docs:
            answer_parts.append(f"According to internal specs: {docs[0]['content']} [INTERNAL: {docs[0]['id']}].")

    if need_web:
        trace.append({"step": "Tool Call: Web Search", "detail": "Queried live web for recent external API changes."})
        web_res = tool_search_web(message, max_results=2)
        sources.extend(web_res)
        if web_res:
            answer_parts.append(f"External reports note: {web_res[0]['snippet']} [WEB: {web_res[0]['url']}].")

    if not answer_parts:
        trace.append({"step": "Tool Call: Internal Docs (Default)", "detail": "Defaulted to internal search."})
        docs = tool_search_internal_docs(message)
        sources.extend(docs)
        if docs:
            answer_parts.append(f"Internal documentation specifies: {docs[0]['content']} [INTERNAL: {docs[0]['id']}].")

    final_text = " ".join(answer_parts) if answer_parts else "No matching information found across internal and external sources."
    audit = audit_and_verify_citations(final_text, sources)

    return {
        "answer": final_text,
        "trace": trace,
        "sources": sources,
        "audit": audit
    }

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Agentic RAG with Source Provenance</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    .source-internal { border-left: 4px solid #3b82f6; }
    .source-web { border-left: 4px solid #10b981; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-10 w-10 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-xl shadow-lg shadow-indigo-500/20">
        <i class="fa-solid fa-brain"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold">Agentic RAG Engine</h1>
        <p class="text-xs text-slate-400">Autonomous Routing &bull; Internal vs Web &bull; Verified Citations</p>
      </div>
    </div>
    <div class="flex items-center space-x-3">
      <input id="apiKeyInput" type="password" placeholder="OpenAI Key (Leave blank for Demo Mode)" 
        class="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 w-64 focus:outline-none focus:border-indigo-500 transition">
      <button onclick="toggleDocModal()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-2 rounded-lg transition flex items-center gap-1.5">
        <i class="fa-solid fa-folder-plus"></i> Manage Knowledge Base
      </button>
    </div>
  </header>

  <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div class="lg:col-span-2 flex flex-col bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      <div id="chatBox" class="flex-1 p-4 md:p-6 overflow-y-auto space-y-6 max-h-[70vh]">
        <div class="flex items-start gap-3">
          <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-sm"><i class="fa-solid fa-robot"></i></div>
          <div class="bg-slate-800/80 rounded-2xl rounded-tl-none p-4 max-w-[85%] text-sm leading-relaxed border border-slate-700/60">
            Welcome to the <strong>Agentic RAG system</strong>. I can query internal documentation, search the web, or both.
            <div class="mt-3 flex flex-wrap gap-2">
              <button onclick="sendPreset('What does our internal documentation say about OAuth rate limits?')" 
                class="text-xs bg-slate-700 hover:bg-slate-600 text-indigo-200 px-2.5 py-1.5 rounded border border-slate-600 transition">
                🏢 Internal Only Query
              </button>
              <button onclick="sendPreset('What is the latest version of Stripe API released and does it break webhooks?')" 
                class="text-xs bg-slate-700 hover:bg-slate-600 text-emerald-200 px-2.5 py-1.5 rounded border border-slate-600 transition">
                🌐 Web Only Query
              </button>
              <button onclick="sendPreset('What does our Stripe documentation say, and has the API changed recently on the public web?')" 
                class="text-xs bg-slate-700 hover:bg-slate-600 text-amber-200 px-2.5 py-1.5 rounded border border-slate-600 transition">
                ⚡ Hybrid Query (Internal + Web)
              </button>
            </div>
          </div>
        </div>
      </div>

      <div class="p-4 border-t border-slate-800 bg-slate-900/90">
        <form id="chatForm" onsubmit="handleSend(event)" class="flex gap-2">
          <input id="userInput" type="text" placeholder="Ask a question requiring internal docs, web, or both..." 
            class="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 transition">
          <button type="submit" id="sendBtn" class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-3 rounded-xl transition flex items-center gap-2">
            <span>Send</span>
            <i class="fa-solid fa-paper-plane text-xs"></i>
          </button>
        </form>
      </div>
    </div>

    <div class="flex flex-col gap-6">
      <div class="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex flex-col flex-1">
        <h2 class="text-sm font-semibold tracking-wider uppercase text-slate-400 mb-3 flex items-center gap-2">
          <i class="fa-solid fa-network-wired text-indigo-400"></i> Agent Thought & Route Trace
        </h2>
        <div id="reasoningTrace" class="flex-1 space-y-3 overflow-y-auto max-h-[35vh] text-xs font-mono text-slate-300">
          <div class="text-slate-500 italic">No action taken yet. Submit a query to inspect agent tools.</div>
        </div>
      </div>

      <div class="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex flex-col flex-1">
        <h2 class="text-sm font-semibold tracking-wider uppercase text-slate-400 mb-3 flex items-center gap-2">
          <i class="fa-solid fa-shield-halved text-emerald-400"></i> Provenance & Citations
        </h2>
        <div id="sourcesContainer" class="flex-1 space-y-3 overflow-y-auto max-h-[35vh] text-xs">
          <div class="text-slate-500 italic">Retrieved chunks and web results will appear here.</div>
        </div>
      </div>
    </div>
  </main>

  <div id="docModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm hidden items-center justify-center p-4 z-50">
    <div class="bg-slate-900 border border-slate-800 w-full max-w-xl rounded-2xl p-6 shadow-2xl">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-lg font-bold">Add Internal Documentation</h3>
        <button onclick="toggleDocModal()" class="text-slate-400 hover:text-white"><i class="fa-solid fa-xmark"></i></button>
      </div>
      <form onsubmit="handleDocUpload(event)" class="space-y-4 text-xs">
        <div>
          <label class="block text-slate-300 mb-1">Doc ID</label>
          <input id="docId" required placeholder="e.g. DOC-SEC-004" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5">
        </div>
        <div>
          <label class="block text-slate-300 mb-1">Document Title</label>
          <input id="docTitle" required placeholder="e.g. Zero-Trust Access Rules" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5">
        </div>
        <div>
          <label class="block text-slate-300 mb-1">Content</label>
          <textarea id="docContent" rows="4" required placeholder="Enter confidential internal policy text..." class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5"></textarea>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button type="button" onclick="toggleDocModal()" class="px-4 py-2 bg-slate-800 rounded-lg">Cancel</button>
          <button type="submit" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-semibold">Index Document</button>
        </div>
      </form>
    </div>
  </div>

  <script>
    function toggleDocModal() {
      const modal = document.getElementById('docModal');
      modal.classList.toggle('hidden');
      modal.classList.toggle('flex');
    }

    function sendPreset(text) {
      document.getElementById('userInput').value = text;
      document.getElementById('chatForm').dispatchEvent(new Event('submit'));
    }

    async function handleDocUpload(e) {
      e.preventDefault();
      const payload = {
        id: document.getElementById('docId').value,
        title: document.getElementById('docTitle').value,
        content: document.getElementById('docContent').value
      };
      await fetch('/api/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      alert('Internal Document indexed successfully!');
      toggleDocModal();
      document.getElementById('docId').value = '';
      document.getElementById('docTitle').value = '';
      document.getElementById('docContent').value = '';
    }

    async function handleSend(e) {
      e.preventDefault();
      const input = document.getElementById('userInput');
      const query = input.value.trim();
      const apiKey = document.getElementById('apiKeyInput').value.trim();
      if (!query) return;

      const chatBox = document.getElementById('chatBox');
      const sendBtn = document.getElementById('sendBtn');

      chatBox.innerHTML += `
        <div class="flex items-start justify-end gap-3">
          <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none p-4 max-w-[85%] text-sm">
            ${escapeHtml(query)}
          </div>
        </div>
      `;
      input.value = '';
      sendBtn.disabled = true;
      sendBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: query, api_key: apiKey || null })
        });
        const data = await res.json();
        const formattedAnswer = marked.parse(data.answer);

        chatBox.innerHTML += `
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-sm shrink-0">
              <i class="fa-solid fa-robot"></i>
            </div>
            <div class="bg-slate-800/80 rounded-2xl rounded-tl-none p-4 max-w-[85%] text-sm leading-relaxed border border-slate-700/60">
              ${formattedAnswer}
              <div class="mt-2 text-xs flex items-center gap-2 text-slate-400">
                <span class="inline-flex items-center gap-1"><i class="fa-solid fa-shield-check text-emerald-400"></i> Citations Validated: ${data.audit.is_safe ? 'Safe' : 'Unverified Detected'}</span>
              </div>
            </div>
          </div>
        `;

        renderTrace(data.trace);
        renderSources(data.sources, data.audit);
      } catch (err) {
        chatBox.innerHTML += `
          <div class="p-3 bg-red-950 border border-red-800 text-red-300 rounded-lg text-xs">
            Error: ${err.message}
          </div>
        `;
      } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = `<span>Send</span> <i class="fa-solid fa-paper-plane text-xs"></i>`;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    function renderTrace(trace) {
      const container = document.getElementById('reasoningTrace');
      container.innerHTML = trace.map(t => `
        <div class="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
          <div class="text-indigo-400 font-bold mb-1">${t.step}</div>
          <div class="text-slate-300">${escapeHtml(t.detail)}</div>
        </div>
      `).join('');
    }

    function renderSources(sources, audit) {
      const container = document.getElementById('sourcesContainer');
      if (!sources || sources.length === 0) {
        container.innerHTML = `<div class="text-slate-500 italic">No external sources queried.</div>`;
        return;
      }

      container.innerHTML = sources.map(s => {
        const isInternal = s.source_type === 'INTERNAL';
        return `
          <div class="p-3 bg-slate-950 rounded-lg border border-slate-800 ${isInternal ? 'source-internal' : 'source-web'}">
            <div class="flex items-center justify-between mb-1">
              <span class="font-bold text-[11px] ${isInternal ? 'text-blue-400' : 'text-emerald-400'}">
                ${isInternal ? '🏢 INTERNAL DOC: ' + s.id : '🌐 WEB: ' + s.title}
              </span>
            </div>
            <p class="text-slate-400 text-[11px] line-clamp-3">${escapeHtml(s.content || s.snippet)}</p>
            ${!isInternal ? `<a href="${s.url}" target="_blank" class="text-indigo-400 underline text-[10px] mt-1 block truncate">${s.url}</a>` : ''}
          </div>
        `;
      }).join('');
    }

    function escapeHtml(string) {
      return String(string).replace(/[&<>"']/g, function (s) {
        return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[s];
      });
    }
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)