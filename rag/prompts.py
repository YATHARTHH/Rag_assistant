import os
import json
import logging
from typing import List, Dict

logger = logging.getLogger("rag_api")

def get_system_prompt(style: str) -> str:
    prompts_path = "./prompts.json"
    default_prompt = "You are a helpful AI assistant."
    if not os.path.exists(prompts_path):
        return default_prompt
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(style, default_prompt)
    except Exception:
        return default_prompt

def rewrite_query_with_history(query: str, chat_history: List[Dict], llm) -> str:
    """
    Reformulates follow-up queries using chat history to make them search-friendly.
    """
    if not chat_history:
        return query
        
    history_str = ""
    for msg in chat_history[-5:]:  # Analyze last 5 turns to stay fast
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    prompt = f"""
    Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be searched in a database.
    Do NOT answer the question. Just output the rewritten standalone question and nothing else.

    Conversation History:
    {history_str}

    Follow-up Question: {query}
    Standalone Question:
    """
    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten:
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten
    except Exception as e:
        logger.warning(f"[PROMPTS] Error rewriting query: {e}")
    return query

def generate_stepback_query(query: str, llm) -> str:
    """
    Generates a broader, abstracted version of the query for wider context retrieval (Step-Back Prompting).
    """
    prompt = f"""You are an expert at abstracting specific questions into broader conceptual ones.
Given the specific query below, generate a single more general/abstract question that covers the underlying concept.

Specific query: "{query}"
Broader abstract question (respond with ONLY the broader question, no explanation):"""
    try:
        response = llm.invoke(prompt)
        broader = response.content.strip()
        if broader:
            return broader
    except Exception as e:
        logger.warning(f"[PROMPTS] Error generating stepback query: {e}")
    return query

def detect_multi_hop_query(query: str, llm) -> bool:
    """
    Detects if a query requires combining facts from multiple documents (multi-hop reasoning).
    """
    prompt = f"""Determine if answering this question requires finding and combining facts from multiple different sources or documents.
Question: "{query}"
Respond with exactly one word: 'yes' or 'no'."""
    try:
        response = llm.invoke(prompt)
        verdict = response.content.strip().lower()
        verdict = "".join([c for c in verdict if c.isalnum()])
        return verdict == "yes"
    except Exception as e:
        logger.warning(f"[PROMPTS] Error detecting multi-hop: {e}")
    return False

def classify_query_intent(query: str, llm) -> str:
    """
    Classifies the user prompt intent into conversational, general, or RAG.
    """
    prompt = f"""
    Classify the following user query into exactly one of these categories:
    - 'conversational' (for greetings, farewells, casual small talk, thanks, or self-introductions)
    - 'general' (for general programming questions, broad math/science questions, writing tasks, or logic puzzles that do NOT refer to specific uploaded papers/documents)
    - 'rag' (for questions asking about uploaded research papers, documents, data, stats, or specific files)

    User Query: "{query}"

    Respond with ONLY one word from: ['conversational', 'general', 'rag']. Do NOT include any punctuation or extra text.
    """
    try:
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        intent = "".join([c for c in intent if c.isalnum()])
        if intent in ["conversational", "general", "rag"]:
            return intent
    except Exception as e:
        logger.warning(f"[PROMPTS] Error classifying intent: {e}")
    return "rag"  # Fallback to RAG query
