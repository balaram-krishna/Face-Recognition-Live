import gradio as gr
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import math, time, os, zipfile, urllib.request

# -------------------------
# Ensure model is available
# -------------------------
MODEL_DIR = "./models/buffalo_s"
if not os.path.exists(MODEL_DIR):
    os.makedirs("./models", exist_ok=True)
    print("📥 Downloading buffalo_s model...")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
    zip_path = "./models/buffalo_s.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./models")
    os.remove(zip_path)
    print("✅ Model downloaded and extracted.")

# -------------------------
# Initialize detector
# -------------------------
DETECTOR_DET_SIZE = (1024, 1024)
detector = FaceAnalysis(name="buffalo_s", root="./models")
detector.prepare(ctx_id=-1, det_size=DETECTOR_DET_SIZE)
print("✅ Detector ready. det_size:", DETECTOR_DET_SIZE)

# -------------------------
# Utilities
# -------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    a = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    b = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = a + b - interArea
    return interArea / denom if denom > 0 else 0.0

def nms(boxes, scores=None, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    if scores is None:
        scores = [1.0] * len(boxes)
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

# -------------------------
# Rotation + multi-scale detection
# -------------------------
def rotate_image_and_mapper(img, angle):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Minv = cv2.invertAffineTransform(M)

    def map_box_back(box):
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                            [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        orig = corners @ Minv.T
        xs, ys = orig[:, 0], orig[:, 1]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return rotated, map_box_back

def detect_augmented(img_rgb,
                     angles=[-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90],
                     scales=[1.0, 1.3],
                     det_thresh=0.20,
                     iou_nms=0.22):
    H, W = img_rgb.shape[:2]
    all_boxes = []
    for scale in scales:
        img_s = cv2.resize(img_rgb, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img_rgb
        for angle in angles:
            rot, back = rotate_image_and_mapper(img_s, angle)
            faces = detector.get(rot)
            for f in faces:
                score = float(getattr(f, "det_score", 1.0))
                if score < det_thresh:
                    continue
                bx = f.bbox.astype(int).tolist()
                mapped = back(bx)
                if scale != 1.0:
                    mapped = [int(mapped[0]/scale), int(mapped[1]/scale),
                              int(mapped[2]/scale), int(mapped[3]/scale)]
                mapped = [max(0, min(W-1, mapped[0])),
                          max(0, min(H-1, mapped[1])),
                          max(0, min(W, mapped[2])),
                          max(0, min(H, mapped[3]))]
                all_boxes.append({'bbox': mapped, 'score': score})
    raw_boxes = [b['bbox'] for b in all_boxes]
    scores = [b['score'] for b in all_boxes]
    keep = nms(raw_boxes, scores, iou_thresh=iou_nms)
    return [all_boxes[i] for i in keep]

# -------------------------
# Embedding extraction
# -------------------------
def crop_and_get_embedding(img_rgb, bbox):
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [max(0, min(w-1, int(round(x)))) if i % 2 == 0
                      else max(0, min(h-1, int(round(x)))) for i, x in enumerate(bbox)]
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = img_rgb[y1:y2, x1:x2]
    faces = detector.get(crop)
    if len(faces) == 0:
        return None
    f = faces[0]
    emb = getattr(f, "normed_embedding", None)
    if emb is None:
        emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-10)
    return emb

# -------------------------
# Matching logic
# -------------------------
def greedy_match_unique(embs1, embs2, sim_thresh=0.38):
    pairs = []
    for i, e1 in enumerate(embs1):
        for j, e2 in enumerate(embs2):
            if e1 is None or e2 is None:
                continue
            sim = float(np.dot(e1, e2))
            pairs.append((i, j, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    matched1, matched2, matches = set(), set(), []
    for i, j, sim in pairs:
        if sim < sim_thresh:
            break
        if i in matched1 or j in matched2:
            continue
        matched1.add(i)
        matched2.add(j)
        matches.append((i, j, sim))
    return matches

# -------------------------
# Comparison Function
# -------------------------
def compare_two_groups(img1_pil, img2_pil,
                       det_threshold=0.20,
                       sim_threshold=0.38,
                       do_multiscale=True,
                       angles_str="-90,-60,-45,-30,-15,0,15,30,45,60,90"):
    angles = [int(x) for x in angles_str.split(",") if x.strip()]
    scales = [1.0, 1.4] if do_multiscale else [1.0]

    if img1_pil is None or img2_pil is None:
        return "Upload both images", None, None

    t0 = time.time()
    img1 = np.array(img1_pil.convert("RGB"))
    img2 = np.array(img2_pil.convert("RGB"))

    dets1 = detect_augmented(img1, angles=angles, scales=scales, det_thresh=det_threshold)
    dets2 = detect_augmented(img2, angles=angles, scales=scales, det_thresh=det_threshold)

    boxes1 = [d['bbox'] for d in dets1]
    boxes2 = [d['bbox'] for d in dets2]
    embs1 = [crop_and_get_embedding(img1, b) for b in boxes1]
    embs2 = [crop_and_get_embedding(img2, b) for b in boxes2]

    matches = greedy_match_unique(embs1, embs2, sim_thresh=sim_threshold)
    matched1 = {m[0] for m in matches}
    matched2 = {m[1] for m in matches}

    out1, out2 = img1.copy(), img2.copy()
    for i, b in enumerate(boxes1):
        x1, y1, x2, y2 = map(int, b)
        color = (255, 255, 0) if i in matched1 else (0, 0, 255)
        cv2.rectangle(out1, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out1, f"{i}", (x1, max(12, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for j, b in enumerate(boxes2):
        x1, y1, x2, y2 = map(int, b)
        color = (0, 255, 0) if j in matched2 else (0, 0, 255)
        cv2.rectangle(out2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out2, ("M" if j in matched2 else "U"),
                    (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    t1 = time.time()
    summary = (f"Group1 faces: {len(boxes1)} | Group2 faces: {len(boxes2)}\n"
               f"Matched pairs: {len(matches)} | Time: {t1-t0:.1f}s")
    return summary, Image.fromarray(out1), Image.fromarray(out2)

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔎 Robust Group-to-Group Face Matching (rotation + multi-scale + fallback)")
    with gr.Row():
        inp1 = gr.Image(type="pil", label="Upload Group Image 1")
        inp2 = gr.Image(type="pil", label="Upload Group Image 2")
    with gr.Row():
        det_thresh = gr.Slider(0.05, 0.5, value=0.20, step=0.01, label="Detection threshold")
        sim_thresh = gr.Slider(0.25, 0.6, value=0.38, step=0.01, label="Similarity threshold")
        multiscale = gr.Checkbox(value=True, label="Enable multi-scale")
    angles_str = gr.Textbox(value="-90,-60,-45,-30,-15,0,15,30,45,60,90", label="Rotation angles")
    btn = gr.Button("Compare Groups")
    out_text = gr.Textbox(label="Summary", interactive=False)
    with gr.Row():
        out1 = gr.Image(label="Annotated Group 1")
        out2 = gr.Image(label="Annotated Group 2")
    btn.click(compare_two_groups,
              inputs=[inp1, inp2, det_thresh, sim_thresh, multiscale, angles_str],
              outputs=[out_text, out1, out2])

# -------------------------
# Render deploy hook
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)

