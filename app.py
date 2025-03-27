from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from cv2 import dnn_superres
import uuid
import base64
import cloudinary
import cloudinary.uploader
import cloudinary.api
from tempfile import NamedTemporaryFile
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Image Enhancement and Detection API")

# Global variable for model
yolo_model = None

@app.on_event("startup")
async def load_model():
    global yolo_model
    import torch
    # Force CPU usage and optimize memory
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🔄 Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")
    # Force model to CPU and half precision
    yolo_model.to('cpu')
    yolo_model.model.half()  # Use half precision
    print("✅ YOLO model loaded successfully!\n")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load ResNet model for species classification
print("🔄 Loading ResNet model for species classification...")
classifier = models.resnet18(weights='IMAGENET1K_V1')  # Using smaller ResNet18 instead of ResNet50
classifier.eval()
classifier.to('cpu')
classifier.half()  # Use half precision
print("✅ ResNet model loaded successfully!\n")

# Define ImageNet class labels (simplified for species mapping)
imagenet_classes = {idx: f"Species_{idx}" for idx in range(1000)}  # Placeholder mapping

# Define preprocessing transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load FSRCNN model for super-resolution
sr = dnn_superres.DnnSuperResImpl_create()
model_path = "FSRCNN_x4.pb"
sr.readModel(model_path)
sr.setModel("fsrcnn", 4)
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Using CPU for Super-Resolution (CUDA disabled)")

# Initialize Cloudinary with URL
cloudinary.config(
    cloud_name="damf0dyl0",
    api_key="121447939383747",
    api_secret="VlJ-sMe8_vwFHl4QK2RZQo3vtFM"
)

# Function to classify species
def classify_species(image_crop):
    with torch.no_grad():  # Reduce memory usage during inference
        image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image_pil).unsqueeze(0).to('cpu').half()  # Use half precision
        output = classifier(input_tensor)
        
    predicted_species = output.argmax().item()
    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_species].item()

    species_name = imagenet_classes.get(predicted_species, "Unknown Species")
    return species_name, confidence

# Function to detect animals and classify species
def detect_and_classify_species(image_path, output_path, conf_threshold=0.3, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA not available. Switching to CPU.\n")
        device = "cpu"

    print(f"🔍 Running YOLO object detection on {image_path} with confidence threshold {conf_threshold}...\n")
    results = yolo_model.predict(source=image_path, conf=conf_threshold, device=device, save=False, show=False)

    detections = results[0].boxes
    annotated_image = results[0].plot()

    detected_species = []
    img = cv2.imread(image_path)

    print(f"✅ Total objects detected: {len(detections)}\n")

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0].item())  # YOLO class ID
        class_name = yolo_model.names[class_id]  # Class label (e.g., "dog", "cat")
        confidence = box.conf[0].item()  # Confidence score

        print(f"🔹 Object {i+1}: {class_name}")
        print(f"   - Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"   - Detection Confidence: {confidence:.2f}")

        animal_crop = img[y1:y2, x1:x2]
        if animal_crop.size == 0:
            print("   ⚠️ Skipping empty crop.\n")
            continue  # Skip if crop is empty

        species_name, species_conf = classify_species(animal_crop)
        detected_species.append(f"{class_name} → {species_name} (Conf: {species_conf:.2f})")

        print(f"   - Classified as: {species_name}")
        print(f"   - Classification Confidence: {species_conf:.2f}\n")

    cv2.imwrite(output_path, annotated_image)

    return detected_species

# Function to check if an image is already enhanced
def is_already_enhanced(image, sharpness_threshold=200, contrast_threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.max() - gray.min()

    print(f"🔍 **Sharpness Score:** {sharpness:.2f} | **Contrast Score:** {contrast:.2f}")

    return sharpness > sharpness_threshold and contrast > contrast_threshold

# Function to enhance an image
def enhance_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image.")

    if is_already_enhanced(image):
        print("✅ **Input image is already enhanced. Skipping Super-Resolution.**")
        cv2.imwrite(output_path, image)
    else:
        upscaled_image = sr.upsample(image)
        cv2.imwrite(output_path, upscaled_image)
        print("✅ **Super-Resolution Applied Successfully!**")

    print(f"\n📂 **Image saved to:** {output_path}")

# Combined endpoint for enhancement and detection
@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # Create temporary files
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.file.read())
            file_path = temp_file.name

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_enhanced:
            enhanced_path = temp_enhanced.name

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_detected:
            detected_path = temp_detected.name

        # Process images
        enhance_image(file_path, enhanced_path)
        detected_species = detect_and_classify_species(enhanced_path, detected_path)

        # Upload to Cloudinary
        enhanced_result = cloudinary.uploader.upload(enhanced_path)
        detected_result = cloudinary.uploader.upload(detected_path)

        # Clean up temporary files
        for path in [file_path, enhanced_path, detected_path]:
            if os.path.exists(path):
                os.remove(path)

        # Return results with Cloudinary URLs
        return JSONResponse(content={
            "enhanced_image_url": enhanced_result["secure_url"],
            "detected_image_url": detected_result["secure_url"],
            "detected_species": detected_species
        })

    except Exception as e:
        # Clean up temporary files in case of error
        for path in [file_path, enhanced_path, detected_path]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))

# Add new endpoints to serve the images
@app.get("/image/{image_path}")
async def get_image(image_path: str):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)