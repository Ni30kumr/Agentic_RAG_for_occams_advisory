import json
from pathlib import Path

def extract_clean_text(input_json, output_txt):
    """Extracts cleaned content from JSON and saves to text file"""
    try:
        # Load JSON data
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare output content
        output_lines = []
        for idx, page in enumerate(data.get('pages', [])):
            # Extract metadata
            meta = page.get('metadata', {})
            source = meta.get('source', f'Unknown Source {idx+1}')
            content_type = meta.get('content_type', 'Unknown Type')
            
            # Get cleaned content
            content = page.get('cleaned_content', '').strip()
            
            # Format entry
            output_lines.append(f"# Source: {source}")
            output_lines.append(f"# Content Type: {content_type}")
            output_lines.append(content)
            output_lines.append("\n" + "="*80 + "\n")
        
        # Write to text file
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            
        print(f"Successfully extracted {len(data['pages'])} pages to {output_txt}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # Configuration
    INPUT_JSON = "occams_content.json"
    OUTPUT_TXT = "occams_clean_text.txt"
    
    # Ensure input file exists
    if not Path(INPUT_JSON).exists():
        print(f"Error: Input file {INPUT_JSON} not found")
    else:
        extract_clean_text(INPUT_JSON, OUTPUT_TXT)