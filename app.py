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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Advanced Image Enhancement and Detection API")

# Global variables for models
yolo_model = None
classifier = None
sr = None

@app.on_event("startup")
async def load_models():
    global yolo_model, classifier, sr
    
    # Setup PyTorch for optimal performance
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load YOLO model
    print("🔄 Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to('cpu')
    yolo_model.model.half()  # Use half precision
    print("✅ YOLO model loaded successfully!")

    # Load ResNet model for species classification
    print("🔄 Loading ResNet model for species classification...")
    classifier = models.resnet18(weights='IMAGENET1K_V1')
    classifier.eval()
    classifier.to('cpu')
    classifier.half()  # Use half precision
    print("✅ ResNet model loaded successfully!")

    # Load FSRCNN model for super-resolution
    sr = dnn_superres.DnnSuperResImpl_create()
    model_path = "FSRCNN_x4.pb"
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✅ Super-Resolution model loaded successfully!")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define ImageNet class labels (simplified for species mapping)
imagenet_classes = {idx: f"Species_{idx}" for idx in range(1000)}

# Define preprocessing transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize Cloudinary with URL
cloudinary.config(
    cloud_name="damf0dyl0",
    api_key="121447939383747",
    api_secret="VlJ-sMe8_vwFHl4QK2RZQo3vtFM"
)

def resize_image(image, max_size=1024):
    """
    Resize image maintaining aspect ratio if it exceeds max_size
    """
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def classify_species(image_crop):
    """
    Classify species using ResNet with improved error handling
    """
    try:
        with torch.no_grad():
            # Resize crop to match ResNet input requirements
            resized_crop = cv2.resize(image_crop, (224, 224))
            
            image_pil = Image.fromarray(cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu').half()
            
            output = classifier(input_tensor)
            
        predicted_species = output.argmax().item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_species].item()

        species_name = imagenet_classes.get(predicted_species, "Unknown Species")
        return species_name, confidence
    except Exception as e:
        logger.error(f"Species classification error: {e}")
        return "Classification Failed", 0.0

def is_already_enhanced(image, sharpness_threshold=150, contrast_threshold=15):
    """
    Check if image is already enhanced with more flexible thresholds
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.max() - gray.min()

    logger.info(f"Sharpness Score: {sharpness:.2f} | Contrast Score: {contrast:.2f}")
    return sharpness > sharpness_threshold and contrast > contrast_threshold

def enhance_image(image_path, output_path):
    """
    Enhanced image processing with resizing and optional super-resolution
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image.")

    # Resize image if too large
    resized_image = resize_image(image)

    if is_already_enhanced(resized_image):
        logger.info("Input image is already enhanced. Skipping Super-Resolution.")
        cv2.imwrite(output_path, resized_image)
    else:
        upscaled_image = sr.upsample(resized_image)
        cv2.imwrite(output_path, upscaled_image)
        logger.info("Super-Resolution Applied Successfully!")

    logger.info(f"Image saved to: {output_path}")

def detect_and_classify_species(image_path, output_path, conf_threshold=0.3):
    """
    Comprehensive object detection and species classification
    """
    detection_logs = []
    detected_species = []

    logger.info(f"Running YOLO object detection on {image_path}")
    results = yolo_model.predict(
        source=image_path, 
        conf=conf_threshold, 
        device='cpu', 
        save=False, 
        show=False,
        verbose=False
    )

    detections = results[0].boxes
    annotated_image = results[0].plot()
    img = cv2.imread(image_path)

    total_detection_log = f"🔍 Total objects detected: {len(detections)}"
    logger.info(total_detection_log)
    detection_logs.append(total_detection_log)

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0].item())
        class_name = yolo_model.names[class_id]
        confidence = box.conf[0].item()

        object_log = [
            f"🔹 Object {i+1}: {class_name}",
            f"   - Bounding Box: ({x1}, {y1}) to ({x2}, {y2})",
            f"   - Detection Confidence: {confidence:.2f}"
        ]
        
        detection_logs.extend(object_log)
        logger.info(object_log[0])
        logger.info(object_log[1])
        logger.info(object_log[2])

        animal_crop = img[y1:y2, x1:x2]
        if animal_crop.size == 0:
            skipped_log = "   ⚠️ Skipping empty crop."
            detection_logs.append(skipped_log)
            logger.warning(skipped_log)
            continue

        species_name, species_conf = classify_species(animal_crop)
        detected_species_log = f"   - Classified as: {species_name} (Conf: {species_conf:.2f})"
        detection_logs.append(detected_species_log)
        logger.info(detected_species_log)

        detected_species.append({
            'class_name': class_name,
            'species_name': species_name,
            'confidence': float(species_conf),
            'bounding_box': {
                'x1': x1, 'y1': y1, 
                'x2': x2, 'y2': y2
            }
        })

    cv2.imwrite(output_path, annotated_image)
    return detected_species, detection_logs

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # Create temporary files with unique names
        file_path = f"/tmp/{uuid.uuid4()}.jpg"
        enhanced_path = f"/tmp/{uuid.uuid4()}.jpg"
        detected_path = f"/tmp/{uuid.uuid4()}.jpg"

        # Write uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process images
        enhance_image(file_path, enhanced_path)
        detected_species, detection_logs = detect_and_classify_species(enhanced_path, detected_path)

        # Upload to Cloudinary
        enhanced_result = cloudinary.uploader.upload(enhanced_path)
        detected_result = cloudinary.uploader.upload(detected_path)

        # Clean up temporary files
        for path in [file_path, enhanced_path, detected_path]:
            if os.path.exists(path):
                os.remove(path)

        # Return comprehensive results
        return JSONResponse(content={
            "enhanced_image_url": enhanced_result["secure_url"],
            "detected_image_url": detected_result["secure_url"],
            "detected_species": detected_species,
            "detection_logs": detection_logs
        })

    except Exception as e:
        # Clean up temporary files in case of error
        for path in [file_path, enhanced_path, detected_path]:
            if os.path.exists(path):
                os.remove(path)
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint to serve images (optional)
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