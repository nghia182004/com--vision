import os
import shutil

def merge_datasets(train_dir='train', test_dir='test', new_dataset_dir='new_dataset'):
    # Collect all class names from train and test directories
    train_classes = set()
    if os.path.isdir(train_dir):
        train_classes = set([entry for entry in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, entry))])
    
    test_classes = set()
    if os.path.isdir(test_dir):
        test_classes = set([entry for entry in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, entry))])
    
    all_classes = train_classes.union(test_classes)
    
    # Create new dataset directory
    os.makedirs(new_dataset_dir, exist_ok=True)
    
    for class_name in all_classes:
        new_class_dir = os.path.join(new_dataset_dir, class_name)
        os.makedirs(new_class_dir, exist_ok=True)
        
        # Copy images from train directory
        train_class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(train_class_path):
            for file_name in os.listdir(train_class_path):
                src = os.path.join(train_class_path, file_name)
                if os.path.isfile(src):
                    dst = os.path.join(new_class_dir, file_name)
                    shutil.copy2(src, dst)
        
        # Copy images from test directory
        test_class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(test_class_path):
            for file_name in os.listdir(test_class_path):
                src = os.path.join(test_class_path, file_name)
                if os.path.isfile(src):
                    dst = os.path.join(new_class_dir, file_name)
                    shutil.copy2(src, dst)
    print("Dataset merging completed successfully!")

# Example usage
merge_datasets(train_dir='train', test_dir='test', new_dataset_dir='new_dataset')