import fitz

def extract_pdf_structured(pdf_path):
    doc = fitz.open(pdf_path)
    
    all_spans = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for b in blocks:
            if "lines" not in b:
                continue
                
            for l in b["lines"]:
                for s in l["spans"]:
                    text = s["text"].strip()
                    if not text:
                        continue
                        
                    all_spans.append({
                        "text": text,
                        "size": s["size"],
                        "font": s["font"],
                        "bbox": s["bbox"],
                        "page": page_num
                    })

                                    
    sizes = sorted({s["size"] for s in all_spans}, reverse=True)
    size_rank = {size: i for i, size in enumerate(sizes)}

                                                              
    num_sizes = len(sizes)
    if num_sizes == 1:
        header_threshold = 0
    elif num_sizes == 2:
        header_threshold = 1
    else:
        header_threshold = 2

                            
    structured_data = []
    for s in all_spans:
        rank = size_rank[s["size"]]
        
        if rank < header_threshold:
            type_val = "header"
        else:
            type_val = "body"
            
        structured_data.append({
            "type": type_val,
            "level": rank + 1,
            "content": s["text"]
        })

    return structured_data

