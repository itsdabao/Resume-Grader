import fitz
import logging
import os
from app.core.bootstrap import bootstrap_runtime
from app.services.cv_parser import extract_text_from_pdf

bootstrap_runtime()
logging.basicConfig(level=logging.INFO)

doc = fitz.open()
page = doc.new_page()
doc.save('test_scan.pdf')
doc.close()

b = open('test_scan.pdf', 'rb').read()
res = extract_text_from_pdf(b)
print('RESULT LEN:', len(res))
print('RESULT:', res)
