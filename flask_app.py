# =============================
# 0️⃣ Dependencies are now in requirements.txt
# =============================
import os, io, csv, json, time, hashlib, asyncio, aiohttp, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from ddgs import DDGS
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, render_template, request, redirect, url_for, flash

# =============================
# 1️⃣ Config (Modified for Railway)
# =============================
# Railway Volumes are mounted at a specific path, we'll use /data
# This ensures our data persists across deployments.
DATA_ROOT = "/data"
DATASET_DIR = os.path.join(DATA_ROOT, "face_dataset")
UPLOAD_FOLDER = os.path.join(DATA_ROOT, "uploads")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_IMAGES_PER_QUERY = 20
SIMILARITY_THRESHOLD = 0.4

# Initialize face analysis model
# This will run when the container starts. It might increase startup time.
print("[*] Initializing FaceAnalysis model...")
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[✓] FaceAnalysis model ready.")

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# It's better to get the secret key from an environment variable on Railway
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key-for-local-dev')

# =============================
# 2️⃣ Enrollment (Unchanged)
# =============================
def enroll_person(name, reference_image_paths):
    embeddings = []
    for path in reference_image_paths:
        try:
            img = np.array(Image.open(path).convert("RGB"))
            faces = face_app.get(img)
            if not faces:
                print(f"[!] No face found in {path}")
                continue
            embeddings.append(faces[0].normed_embedding)
        except Exception as e:
            print(f"[!] Skipping {path}: {e}")
    if not embeddings:
        raise ValueError("No valid reference embeddings found from uploaded images.")
    mean_emb = np.mean(embeddings, axis=0)
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    np.save(os.path.join(person_dir, "enrollment.npy"), mean_emb)
    return mean_emb

# =============================
# 3️⃣ DuckDuckGo Search (Unchanged)
# =============================
def image_search_ddg(query_list, max_results=20, delay=1.5):
    urls = []
    # Using 'with' is important for proper session management
    with DDGS() as ddgs:
        for q in query_list:
            try:
                results = ddgs.images(
                    q, max_results=max_results, safesearch="Off",
                    region="wt-wt", type_image="Photo"
                )
                for r in results:
                    u = r.get("image")
                    if u:
                        urls.append(u)
                time.sleep(delay)
            except Exception as e:
                print(f"[!] Search failed for '{q}': {e}")
    return list(dict.fromkeys(urls))

# =============================
# 4️⃣ Async Downloader (Unchanged)
# =============================
async def fetch_image(session, url, timeout=20):
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200: return None
            return await resp.read()
    except Exception:
        return None

async def download_images(urls, concurrency=10):
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        sem = asyncio.Semaphore(concurrency)
        async def _dl(u):
            async with sem:
                return await fetch_image(session, u)
        tasks = [(_dl(u), u) for u in urls]
        # Return tuples of (data, url) to keep them associated
        results = await asyncio.gather(*(t[0] for t in tasks))
        return [(res, tasks[i][1]) for i, res in enumerate(results)]

def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()

# =============================
# 5️⃣ Build Dataset (Slightly modified to pass URL)
# =============================
async def build_dataset_for_person(name, enrollment_emb, queries):
    person_dir = os.path.join(DATASET_DIR, name)
    meta_path = os.path.join(person_dir, "metadata.csv")
    fieldnames = ["filename", "source_url", "box", "confidence", "distance", "timestamp"]
    with open(meta_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    urls = image_search_ddg(queries, max_results=MAX_IMAGES_PER_QUERY)
    print(f"[+] Found {len(urls)} candidate image URLs for {name}")

    downloads_with_urls = await download_images(urls, concurrency=12)

    saved = 0
    for data, url in tqdm(downloads_with_urls, desc=f"Processing images for {name}"):
        if not data or len(data) < 1000: continue
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except:
            continue

        faces = face_app.get(img)
        for i, face in enumerate(faces):
            emb = face.normed_embedding
            dist = cosine(emb, enrollment_emb)
            if dist <= SIMILARITY_THRESHOLD:
                ts = int(time.time())
                fname = f"{sha1_bytes(data)}_{i}.jpg"
                path = os.path.join(person_dir, fname)

                h, w = img.shape[:2]
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1: continue

                Image.fromarray(img[y1:y2, x1:x2]).save(path, "JPEG", quality=85)

                with open(meta_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        "filename": fname, "source_url": url,
                        "box": json.dumps(face.bbox.tolist()),
                        "confidence": float(face.det_score), "distance": float(dist),
                        "timestamp": ts
                    })
                saved += 1
    print(f"[✓] Saved {saved} verified face crops for {name} → {person_dir}")
    return saved

# =============================
# 🔹 Flask Routes (Modified for temp dir in /data)
# =============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    person_name = request.form.get('person_name', '').strip()
    if not person_name:
        flash('Error: Person name is required.')
        return redirect(url_for('index'))

    files = request.files.getlist('reference_images')
    if not files or all(f.filename == '' for f in files):
        flash('Error: At least one reference image is required.')
        return redirect(url_for('index'))

    # Use a temporary directory inside our persistent /data volume
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{person_name}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    reference_paths = []
    for file in files:
        if file:
            path = os.path.join(temp_dir, file.filename)
            file.save(path)
            reference_paths.append(path)
    
    try:
        enrollment_emb = enroll_person(person_name, reference_paths)
        queries = [
            person_name, f"{person_name} headshot", f"{person_name} closeup",
            f"{person_name} face", f"{person_name} selfie", f"{person_name} smiling"
        ]
        saved_count = asyncio.run(build_dataset_for_person(person_name, enrollment_emb, queries))
        flash(f"Success! Saved {saved_count} verified images for {person_name}. You can find them in the attached volume.", "success")
    except Exception as e:
        print(f"[!!!] An error occurred: {e}")
        flash(f"An error occurred: {e}", "error")
    finally:
        # Clean up the temporary upload directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return redirect(url_for('index'))
