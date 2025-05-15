import shutil
import random
from pathlib import Path

def organize_dataset():
    """Organize dataset into YOLO format with 80/10/10 split, excluding Black Rot"""
    base_dir = Path('C:/final project/grape_leap_detector')
    data_dir = base_dir / 'data'
    
    # Limit number of images per class for faster training
    max_images_per_class = 2000  # <-- Set this to your desired limit
    
    # Create required directories
    train_img_dir = data_dir / 'train' / 'images'
    train_label_dir = data_dir / 'train' / 'labels'
    test_img_dir = data_dir / 'test' / 'images'
    test_label_dir = data_dir / 'test' / 'labels'
    val_img_dir = data_dir / 'val' / 'images'
    val_label_dir = data_dir / 'val' / 'labels'
    
    for dir in [train_img_dir, train_label_dir, test_img_dir, test_label_dir, val_img_dir, val_label_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Only include ESCA, Healthy, Leaf Blight
    classes = {
        'ESCA': 0,
        'Healthy': 1,
        'Leaf Blight': 2
    }
    
    for class_name, class_id in classes.items():
        print(f"\nProcessing {class_name}...")
        source_dir = data_dir / class_name
        if not source_dir.exists():
            print(f"Warning: {class_name} directory not found")
            continue
            
        # Get all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(list(source_dir.glob(ext)))
        
        random.shuffle(images)
        images = images[:max_images_per_class]  # Limit images per class
        
        # Split into train/test/val (80/10/10)
        total_images = len(images)
        train_split = int(total_images * 0.8)
        test_split = int(total_images * 0.9)
        
        train_images = images[:train_split]
        test_images = images[train_split:test_split]
        val_images = images[test_split:]
        
        print(f"Total images: {total_images}")
        print(f"Training images: {len(train_images)}")
        print(f"Testing images: {len(test_images)}")
        print(f"Validation images: {len(val_images)}")
        
        # Process training images
        for img in train_images:
            shutil.copy(img, train_img_dir / img.name)
            with open(train_label_dir / f"{img.stem}.txt", 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        # Process testing images
        for img in test_images:
            shutil.copy(img, test_img_dir / img.name)
            with open(test_label_dir / f"{img.stem}.txt", 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
        # Process validation images
        for img in val_images:
            shutil.copy(img, val_img_dir / img.name)
            with open(val_label_dir / f"{img.stem}.txt", 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

if __name__ == "__main__":
    organize_dataset()