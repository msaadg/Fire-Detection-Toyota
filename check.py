from ultralytics import YOLO

# Load the trained model
model = YOLO('trained-models/check-best.pt')

# Check if names are loaded correctly
if model.names:
    print("Model class names:", model.names)
    print("Classes detected by the model:")
    for class_id, class_name in model.names.items():
        print(f"Class ID: {class_id}, Class Name: {class_name}")
else:
    print("Class names are not defined in the model.")