import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from app.services.cv_parser import extract_text_from_pdf

with open(r'd:\IMT_test\AI_Team_25_May.pdf', 'rb') as f:
    b = f.read()

res = extract_text_from_pdf(b)
print('TEXT LENGTH:', len(res))
print('--- TEXT ---')
print(res[:2000])

with open(r'd:\IMT_test\CV_RAG_Agent\scripts\pdf_text.txt', 'w', encoding='utf-8') as f:
    f.write(res)
