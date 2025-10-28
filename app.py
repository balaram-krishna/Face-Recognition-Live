from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# Initialize the face analyzer
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def match_faces(group1_img, group2_img):
    # Convert images to arrays
    img1 = np.array(group1_img.convert("RGB"))
    img2 = np.array(group2_img.convert("RGB"))

    # Detect faces in both groups
    faces1 = app.get(img1)
    faces2 = app.get(img2)

    count1, count2 = len(faces1), len(faces2)

    # Prepare face embeddings
    embeds1 = [f.normed_embedding for f in faces1]
    embeds2 = [f.normed_embedding for f in faces2]

    # Compare and draw rectangles
    img1_out = img1.copy()
    img2_out = img2.copy()

    threshold = 0.36  # lower = stricter

    for i, f1 in enumerate(faces1):
        matched = False
        for j, f2 in enumerate(faces2):
            sim = np.dot(embeds1[i], embeds2[j])
            if sim > threshold:
                matched = True
                break
        color = (0, 255, 0) if matched else (0, 0, 255)
        box = f1.bbox.astype(int)
        cv2.rectangle(img1_out, (box[0], box[1]), (box[2], box[3]), color, 2)

    for j, f2 in enumerate(faces2):
        matched = False
        for i, f1 in enumerate(faces1):
            sim = np.dot(embeds2[j], embeds1[i])
            if sim > threshold:
                matched = True
                break
        color = (0, 255, 0) if matched else (0, 0, 255)
        box = f2.bbox.astype(int)
        cv2.rectangle(img2_out, (box[0], box[1]), (box[2], box[3]), color, 2)

    return (
        Image.fromarray(cv2.cvtColor(img1_out, cv2.COLOR_BGR2RGB)),
        Image.fromarray(cv2.cvtColor(img2_out, cv2.COLOR_BGR2RGB)),
        f"Group 1 Faces: {count1}, Group 2 Faces: {count2}"
    )

iface = gr.Interface(
    fn=match_faces,
    inputs=[
        gr.Image(type="pil", label="Upload Group Photo 1"),
        gr.Image(type="pil", label="Upload Group Photo 2"),
    ],
    outputs=[
        gr.Image(label="Processed Group 1"),
        gr.Image(label="Processed Group 2"),
        gr.Textbox(label="Face Count Summary")
    ],
    title="Live Face Recognition & Matching App",
    description="Uploads two group photos, counts all faces (front, side, partial), and marks matched faces in green and unmatched in red."
)

import os
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
