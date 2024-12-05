from ultralytics import YOLO

# Load the trained model
model = YOLO('trained-models/best-luminous.pt')

# Check if names are loaded correctly
if model.names:
    print("Model class names:", model.names)
else:
    print("Class names are not defined in the model.")
