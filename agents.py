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
            error_msg = str(e).lower()
            if isinstance(e, groq.RateLimitError):
                if "per day" in error_msg or "tpd" in error_msg:
                    print("Daily rate limit hit! Bypassing retry and falling back immediately.")
                else:
                    raise e
            if "token" in error_msg or "context" in error_msg or "limit" in error_msg or "413" in error_msg:
                print(f"Token/Rate limit error encountered. Falling back to llama-3.1-8b-instant...")
                
                if len(prompt) > 15000:
                    truncated_prompt = prompt[:10000] + "\n\n...[Content truncated due to token limits]...\n\n" + prompt[-4500:]
                else:
                    truncated_prompt = prompt
                    
                try:
                    fallback_response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": truncated_prompt}],
                        temperature=0.1
                    )
                    return fallback_response.choices[0].message.content.strip()
                except Exception as inner_e:
                    print(f"Fallback model failed: {inner_e}")
                    return "Error: The document context is too large for the AI to process. Please try a more specific query."
            raise e

    def _compress_tree(self, nodes):
        compressed = []
        for node in nodes:
            entry = {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "summary": node.get("summary")
            }
            if "children" in node and node["children"]:
                entry["children"] = self._compress_tree(node["children"])
            compressed.append(entry)
        return compressed

    def _find_nodes_by_ids(self, nodes, target_ids):

        found = []
        for node in nodes:
            if node.get("node_id") in target_ids:
                found.append(node)
            if "children" in node and node["children"]:
                found.extend(self._find_nodes_by_ids(node["children"], target_ids))
        return found

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

    def navigate(self, query):
        self.trace.clear()
        self.gathered_context.clear()
        
        print(f"\nAgent started one-shot compressed tree search for: '{query}'")
        
        compressed_tree = self._compress_tree([self.tree])
        tree_json = json.dumps(compressed_tree, indent=2)
        
        prompt = f"""
        You are an intelligent document retrieval agent.
        You have been given a structured document tree (in JSON format) containing the titles and summaries of all sections.
        Your goal is to find the exact sections that contain the information needed to answer the user's query.
        
        Document Tree:
        {tree_json}
        
        User Query: "{query}"
        
        Reason about which sections are most relevant. Then, provide the `node_id`s for those sections.
        You MUST return ONLY a valid JSON object with the following structure, and nothing else:
        {{
            "reasoning": "Brief explanation of why you selected these nodes...",
            "node_list": ["root_0_1", "root_1_2"]
        }}
        """
        
        print("Sending compressed tree to LLM...")
        response = self.ask_llm(prompt)
        
        try:
            cleaned_response = response.strip()
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(0)
            elif cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_response).strip()
                
            data = json.loads(cleaned_response)
            target_ids = data.get("node_list", [])
            reasoning = data.get("reasoning", "No reasoning provided.")
            
            self.trace.append(f"Reasoning: {reasoning}")
            
            if not target_ids:
                print("LLM found no relevant nodes.")
                return self.synthesize_final_answer(query)
                
            print(f"LLM selected nodes: {target_ids}")
            
        except json.JSONDecodeError:
            print("Failed to parse LLM JSON response. Defaulting to empty selection.")
            self.trace.append("Failed to parse LLM search reasoning.")
            return self.synthesize_final_answer(query)
            
        matched_nodes = self._find_nodes_by_ids([self.tree], target_ids)
        
        for node in matched_nodes:
            title = node.get("title", "Unknown Section")
            self.trace.append(f"Retrieved Section: {title}")
            
            raw_text = " ".join(node.get("content", [])) if isinstance(node.get("content"), list) else node.get("content", "")
            self.gathered_context.append({
                "source": title,
                "content": raw_text
            })
            
        print("\nNavigation complete. Synthesizing cited answer...")
        return self.synthesize_final_answer(query)