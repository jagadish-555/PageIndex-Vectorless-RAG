import json
import os
import re
import groq
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class NavigationAgent:
    def __init__(self, document_tree):
        self.tree = document_tree
        self.trace = []
        self.gathered_context = [] 

    def _normalize_title(self, title: str) -> str:
        cleaned = title.strip().strip("\"'`“”‘’")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.lower()

    def _fuzzy_match_titles(self, chosen_titles, children):
        title_map = {self._normalize_title(c.get("title", "")): c for c in children}
        matched_nodes = []
        for raw in chosen_titles:
            norm = self._normalize_title(raw)
            node = title_map.get(norm)
            if node:
                matched_nodes.append(node)
                continue

            for key, candidate in title_map.items():
                if norm and (norm in key or key in norm):
                    matched_nodes.append(candidate)
                    break
        return matched_nodes

    def _get_all_text(self, node):
        text = node.get("title", "") + " " + node.get("summary", "") + " "
        if "content" in node:
            if isinstance(node["content"], list):
                text += " ".join(node["content"])
            else:
                text += node["content"]
        if "children" in node:
            for child in node["children"]:
                text += " " + self._get_all_text(child)
        return text

    def _select_fallback_nodes(self, query, children, max_nodes=3):
        stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "about", "of", "what", "where", "when", "why", "how", "does", "did", "do", "it", "this", "that", "these", "those", "from", "by", "as", "be", "has", "have", "had", "will", "would", "can", "could"}
        query_words = set(self._normalize_title(query).split()) - stop_words
        
        def score(child):
            child_text = self._normalize_title(self._get_all_text(child))
            child_words = child_text.split()
            return sum(child_words.count(w) for w in query_words)

        ranked = sorted(children, key=score, reverse=True)
        return [c for c in ranked[:max_nodes] if c]

    @retry(
        retry=retry_if_exception_type(groq.RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def ask_llm(self, prompt, model="llama-3.3-70b-versatile"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if isinstance(e, groq.RateLimitError):
                raise e
            error_msg = str(e).lower()
            if "token" in error_msg or "context" in error_msg or "limit" in error_msg or "413" in error_msg:
                print(f"Token limit error encountered. Falling back to openai/gpt-oss-20b...")
                truncated_prompt = prompt[:15000] + "\n...[Content truncated due to token limits]" if len(prompt) > 15000 else prompt
                try:
                    fallback_response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": truncated_prompt}],
                        temperature=0.1
                    )
                    return fallback_response.choices[0].message.content.strip()
                except Exception as inner_e:
                    print(f"Fallback model failed: {inner_e}")
                    return "Error: The document context is too large for the AI to process. Please try a more specific query."
            raise e

    def choose_paths(self, query, children):
        options_text = ""
        for child in children:
            title = child.get("title", "Unknown Section")
            summary = child.get("summary", "No summary available.")
            options_text += f"- Title: '{title}' | Summary: {summary}\n"

        prompt = f"""
        You are an autonomous agent navigating a structured document to answer a user's query.
        User Query: "{query}"

        Available sub-sections:
        {options_text}

        Which of these sections might contain information relevant to the query?
        Reply ONLY as JSON with this exact shape:
        {{"titles": ["Exact Title 1", "Exact Title 2"]}}
        If none are relevant, reply {{"titles": []}}.
        """
        
        response = self.ask_llm(prompt)
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_response).strip()
                
            data = json.loads(cleaned_response)
            titles = data.get("titles", [])
            if isinstance(titles, list):
                return [t for t in titles if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

        return []

    def synthesize_final_answer(self, query):
        if not self.gathered_context:
            return "I could not find relevant information in the document to answer your query."
            
        context_text = ""
        for item in self.gathered_context:
            context_text += f"\n--- Source: {item['source']} ---\n{item['content']}\n"
            
        prompt = f"""
        User Query: "{query}"
        
        Synthesize a comprehensive answer using ONLY the context provided below. 
        You MUST cite your sources using the [Source Title] format at the end of relevant sentences.
        
        Context:
        {context_text}
        """
        return self.ask_llm(prompt)

    def navigate(self, query, current_node=None, is_root=True):
        if is_root:
            self.trace.clear()
            self.gathered_context.clear()
            print(f"\nAgent started multi-node navigation for: '{query}'")

        if current_node is None:
            current_node = self.tree

        title = current_node.get("title", "Root")
        self.trace.append(title)
        
        if "content" in current_node and (not current_node.get("children")):
            print(f"Extracting context from leaf: '{title}'")
            raw_text = " ".join(current_node["content"]) if isinstance(current_node["content"], list) else current_node.get("content", "")
            
            self.gathered_context.append({
                "source": title,
                "content": raw_text
            })
            if is_root:
                print("\nNavigation complete. Synthesizing cited answer...")
                return self.synthesize_final_answer(query)
            return

        if "children" in current_node and current_node["children"]:
            print(f"Evaluating {len(current_node['children'])} sub-sections under '{title}'...")
            
            chosen_titles = self.choose_paths(query, current_node["children"])
            matched_nodes = self._fuzzy_match_titles(chosen_titles, current_node["children"])
            
            if not matched_nodes:
                print(f"Agent determined no relevant paths under '{title}'.")
                fallback_nodes = self._select_fallback_nodes(query, current_node["children"])
                if fallback_nodes:
                    matched_nodes = fallback_nodes
                else:
                    if is_root:
                        return self.synthesize_final_answer(query)
                    return
                 
            print(f"Agent decided to explore paths: {[n.get('title', 'Unknown') for n in matched_nodes]}")
            
            for next_node in matched_nodes:
                self.navigate(query, next_node, is_root=False)

        if is_root:
            print("\nNavigation complete. Synthesizing cited answer...")
            return self.synthesize_final_answer(query)