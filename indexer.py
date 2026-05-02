import os
import re
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@retry(
    retry=retry_if_exception_type(groq.RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_llm_with_retry(prompt, model="llama-3.3-70b-versatile"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if isinstance(e, groq.RateLimitError):
            raise e  # Let the @retry decorator handle rate limits
            
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
                return "Error: The text is too large to summarize."
        raise e

def generate_summary(text, level_type):
    if not text.strip():
        return "No content to summarize."
        
    prompt = f"Provide a brief, concise summary of the following {level_type} text. This summary will be used for navigating a document tree.\n\nText:\n{text}"
    
    try:
        return call_llm_with_retry(prompt)
    except Exception as e:
        print(f"API Error during summarization: {e}")
        return "Summary failed due to API error."

def get_heuristic_summary(text, num_sentences=2):
    if not text.strip():
        return "Empty section."
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return " ".join(sentences[:num_sentences])

def build_tree(structured_data):
    tree = {"title": "Document", "summary": "", "children": []}

    chapters = []
    current_chapter = None
    current_section = None

    for item in structured_data:
        if item["type"] == "header":
            if item["level"] == 1:
                current_chapter = {"title": item["content"], "children": []}
                chapters.append(current_chapter)
                current_section = None
            else:
                if current_chapter is None:
                    current_chapter = {"title": "Default Chapter", "children": []}
                    chapters.append(current_chapter)
                
                current_section = {"title": item["content"], "content": []}
                current_chapter["children"].append(current_section)

        elif item["type"] == "body":
            if current_chapter is None:
                current_chapter = {"title": "Default Chapter", "children": []}
                chapters.append(current_chapter)
            
            if current_section is None:
                current_section = {"title": "Default Section", "content": []}
                current_chapter["children"].append(current_section)
            else:
                current_length = sum(len(c) for c in current_section["content"])
                if current_length > 3000:
                    base_title = current_section["title"]
                    if not base_title.endswith(" (Continued)"):
                        base_title += " (Continued)"
                    current_section = {"title": base_title, "content": []}
                    current_chapter["children"].append(current_section)
                
            current_section["content"].append(item["content"])

    print("\nStarting High-Speed Summarization Pipeline...")
    
    for chapter in chapters:
        for section in chapter["children"]:
            text = " ".join(section["content"]) if section["content"] else "Empty section."
            section["summary"] = get_heuristic_summary(text) 

        print(f"Asking Groq to summarize Chapter: '{chapter.get('title')}'")
        summaries = " ".join([s.get("summary", "") for s in chapter["children"]])
        chapter["summary"] = generate_summary(summaries, "chapter")

    print(f"Asking Groq to summarize overall Document")
    all_chapter_summaries = " ".join([c.get("summary", "") for c in chapters])
    tree["summary"] = generate_summary(all_chapter_summaries, "document")

    tree["children"] = chapters
    
    def assign_ids(node, prefix="node"):
        if "children" in node:
            for i, child in enumerate(node["children"]):
                child["node_id"] = f"{prefix}_{i}"
                assign_ids(child, child["node_id"])
                
    tree["node_id"] = "root"
    assign_ids(tree, "root")

    return tree