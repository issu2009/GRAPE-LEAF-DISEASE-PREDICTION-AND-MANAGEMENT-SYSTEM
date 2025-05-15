from ultralytics import YOLO
from pathlib import Path
import time

def train_model():
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    current_dir = Path(__file__).parent
    data_yaml_path = str(current_dir / 'data.yaml')

    print("Starting training with faster settings...")
    start_time = time.time()

    try:
        results = model.train(
            data=data_yaml_path,
            epochs=25,           # Reduced epochs
            imgsz=416,           # Reduced image size
            batch=8,             # Keep batch size for RAM safety
            name='grape_disease_model',
            device='cpu',        
            workers=2,
            save=True,
            save_period=5,      
            pretrained=True,
            optimizer='AdamW',
            lr0=0.0005,         
            lrf=0.0001,         
            warmup_epochs=3.0,   
            momentum=0.937,
            weight_decay=0.001,  
            project=str(current_dir / 'runs'),
            exist_ok=True,
            cache=True,
            verbose=True,
            degrees=20.0,        
            translate=0.2,
            scale=0.5,
            fliplr=0.5,
            mosaic=0.7,         
            mixup=0.15,         
            hsv_h=0.015,        
            hsv_s=0.7,
            hsv_v=0.4
        )
        print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes!")
        return results

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__": 
    train_model()