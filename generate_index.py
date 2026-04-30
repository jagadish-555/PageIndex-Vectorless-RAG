import json
import argparse
from pdf_tools import extract_pdf_structured
from indexer import build_tree

def main():
    parser = argparse.ArgumentParser(description="Generate tree_index.json from a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file to index.")
    parser.add_argument("-o", "--output", default="tree_index.json", help="Output JSON file name (default: tree_index.json)")
    args = parser.parse_args()

    print(f"Extracting structured data from {args.pdf_path}...")
    try:
        structured_data = extract_pdf_structured(args.pdf_path)
    except FileNotFoundError:
        print(f"Error: the file '{args.pdf_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    print("Building tree and generating summaries (this might take a minute)...")
    tree = build_tree(structured_data)

    print(f"Saving index to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(tree, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()