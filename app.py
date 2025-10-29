# -------------------------
# Websockets import shim for gradio_client compatibility
# -------------------------
import sys, types, importlib

try:
    import websockets.asyncio  # type: ignore
except Exception:
    try:
        websockets_pkg = importlib.import_module("websockets")
        client_real = None
        try:
            client_real = importlib.import_module("websockets.client")
        except Exception:
            client_real = getattr(websockets_pkg, "client", None)
        if client_real:
            mod_asyncio = types.ModuleType("websockets.asyncio")
            client_mod = types.ModuleType("websockets.asyncio.client")
            for attr in dir(client_real):
                if not attr.startswith("__"):
                    try:
                        setattr(client_mod, attr, getattr(client_real, attr))
                    except Exception:
                        pass
            mod_asyncio.client = client_mod
            sys.modules["websockets.asyncio"] = mod_asyncio
            sys.modules["websockets.asyncio.client"] = client_mod
    except Exception:
        pass

# -------------------------
# Imports
# -------------------------
import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from insightface.app import FaceAnalysis

# -------------------------
# Face Analysis Setup
# -------------------------
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# -------------------------
# Helper Functions
# -------------------------
def get_embedding(img):
    faces = face_app.get(img)
    if not faces:
        return None, 0
    embeddings = [f.embedding for f in faces]
    return embeddings, len(faces)

def compare_faces(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    emb1, c1 = get_embedding(img1)
    emb2, c2 = get_embedding(img2)
    if emb1 is None or emb2 is None:
        return "No face detected in one or both images", None
    scores = [
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        for e1 in emb1 for e2 in emb2
    ]
    s = max(scores)
    result = "✅ Same Person" if s > 0.5 else "❌ Different Persons"
    return f"{result}\nSimilarity: {s:.2f}", s

def group_face_analysis(group_img, single_img):
    group_img = np.array(group_img)
    single_img = np.array(single_img)
    group_emb, gcount = get_embedding(group_img)
    single_emb, scount = get_embedding(single_img)
    if group_emb is None or single_emb is None:
        return "Face not detected in one or both images"
    same = sum(
        np.dot(single_emb[0], g) / (np.linalg.norm(single_emb[0]) * np.linalg.norm(g)) > 0.5
        for g in group_emb
    )
    return f"Total Faces in Group: {gcount}\nMatched Faces: {same}"

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="Face Recognition Live") as demo:
    gr.Markdown("## 👤 Face Recognition App")
    gr.Markdown("Upload images to compare or match faces in a group photo.")
    
    with gr.Tab("Compare Two Faces"):
        i1 = gr.Image(label="Image 1", type="pil")
        i2 = gr.Image(label="Image 2", type="pil")
        out1 = gr.Textbox(label="Result")
        btn1 = gr.Button("Compare Faces")
        btn1.click(compare_faces, [i1, i2], [out1])

    with gr.Tab("Group Face Match"):
        gi = gr.Image(label="Group Image", type="pil")
        si = gr.Image(label="Single Face", type="pil")
        out2 = gr.Textbox(label="Result")
        btn2 = gr.Button("Analyze Group")
        btn2.click(group_face_analysis, [gi, si], [out2])

# -------------------------
# Launch Server
# -------------------------
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
