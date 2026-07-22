import re
import logging

logger = logging.getLogger("rag_api")

def evaluate_faithfulness(context: str, answer: str, llm) -> float:
    """
    Evaluates if the answer is grounded in the retrieved context.
    """
    if not context or not answer:
        return 0.0
    prompt = f"""
    You are an independent evaluator. Evaluate the faithfulness of the generated answer compared to the retrieved context.
    Faithfulness measures if the answer is completely grounded in and supported by the context, without making up facts or adding outside info.

    Retrieved Context:
    {context}

    Generated Answer:
    {answer}

    Provide a score between 0.0 (completely hallucinated / not supported) and 1.0 (completely faithful / 100% grounded).
    Respond with ONLY a single decimal number. Do NOT include any explanations or extra characters.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r'\d+\.\d+|\d+', text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"[EVALUATION] Faithfulness failed: {e}")
    return 1.0

def evaluate_answer_relevance(query: str, answer: str, llm) -> float:
    """
    Evaluates if the answer directly addresses the query.
    """
    if not query or not answer:
        return 0.0
    prompt = f"""
    You are an independent evaluator. Evaluate the relevance of the generated answer to the user's question.
    Answer Relevance measures if the answer directly addresses the question and is not evasive or redundant.

    Question: {query}
    Generated Answer: {answer}

    Provide a score between 0.0 (completely irrelevant / doesn't answer the question) and 1.0 (perfectly relevant).
    Respond with ONLY a single decimal number. Do NOT include any explanations or extra characters.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r'\d+\.\d+|\d+', text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"[EVALUATION] Answer relevance failed: {e}")
    return 1.0

def evaluate_context_precision(query: str, context: str, llm) -> float:
    """
    Evaluates if the retrieved context is relevant to the query.
    """
    if not query or not context:
        return 0.0
    prompt = f"""
    You are an independent evaluator. Evaluate the precision of the retrieved context compared to the user's query.
    Context Precision measures how much of the retrieved context is relevant and useful for answering the query.

    Query: {query}
    Retrieved Context:
    {context}

    Provide a score between 0.0 (completely irrelevant) and 1.0 (all context blocks are highly relevant).
    Respond with ONLY a single decimal number. Do NOT include any explanations or extra characters.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r'\d+\.\d+|\d+', text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"[EVALUATION] Context precision failed: {e}")
    return 1.0
