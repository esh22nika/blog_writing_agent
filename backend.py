from __future__ import annotations

import json
import operator
import re
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama

# ============================================================
# Blog Writer Agent — Ollama / qwen2.5, no external API keys
# Router → Researcher → Orchestrator → Workers → ReducerWithImages
# ============================================================

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should understand.")
    bullets: List[str] = Field(..., description="3-6 concrete sub-points to cover.")
    target_words: int = Field(..., description="Target words 120-550.")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    search_query: str  # used for Unsplash URL (no API key needed)


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# -----------------------------
# 2) LLMs
# -----------------------------
router_llm = ChatOllama(model="qwen2.5", temperature=0)
research_llm = ChatOllama(model="qwen2.5", temperature=0)
plan_llm = ChatOllama(model="qwen2.5", temperature=0)
writer_llm = ChatOllama(model="qwen2.5", temperature=0.3)


# -----------------------------
# Helpers
# -----------------------------
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def _extract_json(text: str) -> str:
    """Extract first {...} or [...] block from text."""
    text = _strip_fences(text)
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    end = text.rfind("}") if "{" in text else text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def _parse_json(text: str) -> dict:
    cleaned = _extract_json(text)
    return json.loads(cleaned)


# -----------------------------
# 3) Router
# -----------------------------
def router_node(state: State) -> dict:
    prompt = f"""You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen, well-known concepts.
- hybrid (needs_research=true): mostly evergreen but benefits from recent examples.
- open_book (needs_research=true): volatile / latest news / weekly roundup.

Topic: {state["topic"]}
As-of date: {state["as_of"]}

Return ONLY valid JSON (no markdown fences):
{{
  "needs_research": true or false,
  "mode": "closed_book" | "hybrid" | "open_book",
  "reason": "...",
  "queries": ["query1", "query2"]
}}"""

    response = router_llm.invoke(prompt).content
    try:
        data = _parse_json(response)
        decision = RouterDecision(**data)
    except Exception:
        decision = RouterDecision(
            needs_research=False, mode="closed_book",
            reason="parse error fallback", queries=[]
        )

    recency_days = 7 if decision.mode == "open_book" else (45 if decision.mode == "hybrid" else 3650)
    print(f"[Router] mode={decision.mode}, needs_research={decision.needs_research}")
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 4) Research (LLM-only, no Tavily)
# -----------------------------
def research_node(state: State) -> dict:
    """Uses qwen2.5 as the knowledge source — no external search API needed."""
    queries_text = "\n".join(f"- {q}" for q in (state.get("queries") or []))

    prompt = f"""You are a knowledgeable research assistant.

The following search queries were generated for a blog about:
"{state["topic"]}"

Queries:
{queries_text}

Based on your knowledge, generate 6-10 concise research notes that would help write this blog.
Each note should be a factual, specific point a writer could cite or expand on.

Return ONLY valid JSON (no markdown fences):
{{
  "evidence": [
    {{
      "title": "Short descriptive title",
      "url": "https://relevant-source-url.com",
      "published_at": null,
      "snippet": "Concise factual note (1-2 sentences)",
      "source": "Source name"
    }}
  ]
}}"""

    response = research_llm.invoke(prompt).content
    try:
        data = _parse_json(response)
        evidence = [EvidenceItem(**e) for e in data.get("evidence", [])]
    except Exception:
        evidence = []

    print(f"[Research] {len(evidence)} evidence items generated")
    return {"evidence": evidence}


# -----------------------------
# 5) Orchestrator
# -----------------------------
def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    evidence_text = "\n".join(
        f"- {e.title}: {e.snippet or ''}" for e in evidence[:12]
    ) or "No evidence."

    prompt = f"""You are a senior technical writer.

Create a detailed blog plan for the topic: {state["topic"]}
Mode: {mode}
As-of: {state["as_of"]}

Research notes:
{evidence_text}

Requirements:
- 5-7 sections
- Each section: title, goal (1 sentence), 3-6 bullets, target_words (120-450)
- Set requires_code=true for sections with code examples
- blog_kind must be one of: explainer, tutorial, news_roundup, comparison, system_design

Return ONLY valid JSON (no markdown fences):
{{
  "blog_title": "...",
  "audience": "...",
  "tone": "...",
  "blog_kind": "explainer",
  "constraints": [],
  "tasks": [
    {{
      "id": 1,
      "title": "...",
      "goal": "...",
      "bullets": ["...", "..."],
      "target_words": 200,
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false
    }}
  ]
}}"""

    response = plan_llm.invoke(prompt).content
    try:
        data = _parse_json(response)
        plan = Plan(**data)
    except Exception as e:
        raise ValueError(f"[Orchestrator] Failed to parse plan: {e}\nRaw: {response[:500]}")

    print(f"[Orchestrator] \"{plan.blog_title}\" — {len(plan.tasks)} sections")
    return {"plan": plan}


# -----------------------------
# 6) Fan-out
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 7) Worker
# -----------------------------
def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title}: {e.snippet or ''}" for e in evidence[:12]
    ) if evidence else "No external evidence."

    prompt = f"""You are a senior technical writer writing ONE section of a blog.

Blog title: {plan.blog_title}
Audience: {plan.audience}
Tone: {plan.tone}
Blog kind: {plan.blog_kind}

Section title: {task.title}
Goal: {task.goal}
Target length: ~{task.target_words} words
Requires code: {task.requires_code}

Cover these points in order:{bullets_text}

Research notes (use as context):
{evidence_text}

{"Include at least one working code snippet." if task.requires_code else ""}

Return clean markdown starting with:
## {task.title}"""

    section_md = writer_llm.invoke(prompt).content.strip()
    print(f"[Worker] Done: {task.title}")
    return {"sections": [(task.id, section_md)]}


# ============================================================
# 8) Reducer subgraph: merge → decide_images → place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


def decide_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None
    merged_md = state["merged_md"]

    prompt = f"""You are a technical editor deciding where to place images in a blog post.

Blog kind: {plan.blog_kind}
Topic: {state["topic"]}

Rules:
- Max 3 images, only where they materially aid understanding (diagrams, flows, concepts).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed, keep md_with_placeholders identical to input and images=[].
- search_query: a short phrase for finding a relevant photo/diagram on Unsplash.

Return ONLY valid JSON (no markdown fences):
{{
  "md_with_placeholders": "<full markdown with placeholders inserted>",
  "images": [
    {{
      "placeholder": "[[IMAGE_1]]",
      "filename": "descriptive_name.jpg",
      "alt": "alt text",
      "caption": "Figure caption",
      "search_query": "neural network attention diagram"
    }}
  ]
}}

Blog markdown:
{merged_md[:3000]}"""

    response = plan_llm.invoke(prompt).content
    try:
        data = _parse_json(response)
        image_plan = GlobalImagePlan(**data)
    except Exception:
        image_plan = GlobalImagePlan(md_with_placeholders=merged_md, images=[])

    print(f"[DecideImages] {len(image_plan.images)} image(s) planned")
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    """Replace placeholders with free Unsplash image URLs — no API key needed."""
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    for spec in image_specs:
        query = spec.get("search_query", spec.get("alt", "technology"))
        query_url = query.replace(" ", ",")
        img_url = f"https://source.unsplash.com/featured/1200x600/?{query_url}"
        img_md = f"![{spec['alt']}]({img_url})\n*{spec['caption']}*"
        md = md.replace(spec["placeholder"], img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    print(f"[Reducer] Saved → {filename}")
    return {"final": md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# -----------------------------
# 9) Main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
