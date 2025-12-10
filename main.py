from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.mount("/", StaticFiles(directory=".", html=True), name="static")

def fast_svd(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


@app.post("/compress")
async def compress(rank: int = Form(...), file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)

    R = fast_svd(arr[:, :, 0], rank)
    G = fast_svd(arr[:, :, 1], rank)
    B = fast_svd(arr[:, :, 2], rank)

    out = np.stack([R, G, B], axis=2).clip(0, 255).astype("uint8")
    out_img = Image.fromarray(out)

    buf = io.BytesIO()
    out_img.save(buf, format="JPEG")

    return {"image": buf.getvalue().hex()}