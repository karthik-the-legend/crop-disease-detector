from pypdf import PdfReader
import glob, os

pdfs = sorted(glob.glob('data/agri_pdfs/*.pdf'))
print(f'Found {len(pdfs)} PDFs\n')

pesticides = ['mancozeb','copper','chlorothalonil','fungicide','dosage','ml/litre']

for pdf in pdfs:
    r = PdfReader(pdf)
    text = ' '.join(p.extract_text() or '' for p in r.pages[:5]).lower()
    hits = [p for p in pesticides if p in text]
    size = round(os.path.getsize(pdf)/1024, 1)
    name = os.path.basename(pdf)
    no_hits = 'NONE -- check PDF content'
    print(f'{name}')
    print(f'  Pages: {len(r.pages)} | Size: {size}KB')
    print(f'  Pesticides found: {hits if hits else no_hits}')
    print()
