import requests, json, glob, os

BASE = 'http://localhost:8000'

print('=' * 55)
print('ENDPOINT TESTS')
print('=' * 55)

# TEST 1: GET /health
r = requests.get(f'{BASE}/health')
d = r.json()
print(f'[1] Health raw: {d}')
print(f'[1] Health: status={d.get("status")} | cnn={d.get("cnn_loaded")} | rag={d.get("rag_loaded")}')

# TEST 2: GET /diseases
r2 = requests.get(f'{BASE}/diseases')
d2 = r2.json()
print(f'[2] Diseases: {d2.get("count")} classes returned')

# TEST 3: GET /sources
r3 = requests.get(f'{BASE}/sources')
d3 = r3.json()
print(f'[3] Sources: {d3.get("count")} PDFs returned')

# TEST 4-7: POST /diagnose in all 4 languages
test_imgs = glob.glob('data/processed/test/**/*.jpg', recursive=True)[:4]
langs = ['en', 'hi', 'te', 'ta']

if not test_imgs:
    print('[4-7] No test images found in data/processed/test/')
else:
    for i, (img_path, lang) in enumerate(zip(test_imgs, langs)):
        with open(img_path, 'rb') as f:
            files = {'image': (os.path.basename(img_path), f, 'image/jpeg')}
            r4 = requests.post(f'{BASE}/diagnose?lang_code={lang}', files=files, timeout=90)
        d4 = r4.json()
        print(f'[{i+4}] Diagnose [{lang}] [{d4.get("severity")}]: {str(d4.get("disease_name",""))[:35]} | {d4.get("latency_ms")}ms')
        print(f'     Treatment: {str(d4.get("treatment",""))[:100]}')

print()
print('=' * 55)
print('EDGE CASE TESTS')
print('=' * 55)

# EDGE 1: Unknown lang_code
if test_imgs:
    with open(test_imgs[0], 'rb') as f:
        files = {'image': (os.path.basename(test_imgs[0]), f, 'image/jpeg')}
        r5 = requests.post(f'{BASE}/diagnose?lang_code=xx', files=files, timeout=90)
    d5 = r5.json()
    print(f'[E1] Unknown lang xx: status={r5.status_code} | lang_code={d5.get("lang_code")}')

# EDGE 2: Wrong file type — use any txt file that exists
dummy_path = os.path.abspath('README.md') if os.path.exists('README.md') else os.path.abspath('requirements.txt')
with open(dummy_path, 'rb') as f:
    r6 = requests.post(f'{BASE}/diagnose', files={'image': ('test.pdf', f, 'application/pdf')}, timeout=30)
print(f'[E2] Wrong file type: status={r6.status_code} (expected: 400)')

print()
print('All endpoint tests complete.')
