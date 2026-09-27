import os
import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # The namespaces in docx xml
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text_list = []
            for paragraph in tree.findall('.//w:p', namespaces):
                para_text = []
                for run in paragraph.findall('.//w:r', namespaces):
                    t = run.find('w:t', namespaces)
                    if t is not None and t.text:
                        para_text.append(t.text)
                text_list.append(''.join(para_text))
            
            return '\n'.join(text_list)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

folder_path = r"c:\Users\chint\BTCUSDT_Quant_System\guideline documents"
output_path = r"c:\Users\chint\BTCUSDT_Quant_System\guideline_contents.md"

with open(output_path, 'w', encoding='utf-8') as f:
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            f.write(f"# {filename}\n")
            f.write(extract_text_from_docx(file_path))
            f.write("\n\n---\n\n")

print("Done extracting text.")
