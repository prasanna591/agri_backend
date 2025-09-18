from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm

router = APIRouter()

# ----------------------------
# Config (edit these paths)
# ----------------------------
WEIGHTS_PATH = "weights.pth"     # put your fine-tuned model weights here
LABELS_PATH = "labels.json"      # JSON list of class names
MODEL_NAME = "efficientnet_b0"   # must match model trained
IMG_SIZE = 224
TOP_K = 3

# ----------------------------
# Load labels
# ----------------------------
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        CLASSES = json.load(f)
else:
    CLASSES = [
        "Healthy Leaf",
        "Leaf Blight",
        "Powdery Mildew",
        "Rust",
        "Leaf Spot",
    ]
NUM_CLASSES = len(CLASSES)

# ----------------------------
# Build + load model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    return model

model = build_model()
if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
else:
    print(f"[WARN] Weights file not found at {WEIGHTS_PATH}, using random init")

model.to(device)
model.eval()

# ----------------------------
# Preprocessing
# ----------------------------
preprocess = T.Compose([
    T.Resize(int(IMG_SIZE * 1.15)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def read_imagefile(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def predict_tensor(input_tensor: torch.Tensor, topk: int = TOP_K):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)
    top_probs = top_probs.cpu().numpy()[0].tolist()
    top_idxs = top_idxs.cpu().numpy()[0].tolist()
    results = [{"label": CLASSES[idx], "confidence": round(float(p), 4)}
               for idx, p in zip(top_idxs, top_probs)]
    return results

# ----------------------------
# Recommendations mapping
# ----------------------------
RECOMMENDATIONS = {
    "Healthy Leaf": ["No action needed. Monitor regularly."],
    "Leaf Blight": ["Remove infected leaves", "Apply approved fungicide", "Avoid overhead irrigation"],
    "Powdery Mildew": ["Use sulfur-based sprays", "Improve airflow"],
    "Rust": ["Remove infected material", "Apply copper fungicide if allowed"],
    "Leaf Spot": ["Improve soil drainage", "Apply targeted fungicide"],
}

def get_recommendations(label: str):
    return RECOMMENDATIONS.get(label, ["Consult local expert", "Collect sample for lab test"])

# ----------------------------
# FastAPI route
# ----------------------------
@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10)
):
    # 1) Read file
    image_bytes = await file.read()
    img = read_imagefile(image_bytes)

    # 2) Preprocess
    input_tensor = preprocess(img).unsqueeze(0)

    # 3) Predict
    try:
        results = predict_tensor(input_tensor, topk=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 4) Build response
    top_result = results[0]
    recs = get_recommendations(top_result["label"])

    return JSONResponse(content={
        "prediction": top_result["label"],
        "confidence": top_result["confidence"],
        "top_k": results,
        "recommendations": recs,
        "image_size": f"{img.width}x{img.height}"
    })
