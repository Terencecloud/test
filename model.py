import os
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from PIL import Image
from sklearn.metrics import accuracy_score

# Define the folder paths for different indices
folder_paths = {
    'SWIR': 'data/Swir'
}

# Define the output folder for processed images and results
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# Database setup
DATABASE_URL = "postgresql://postgres:Smallholder19@localhost/moisture_results"  # Replace with actual credentials
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define SQLAlchemy model for moisture results
class MoistureResult(Base):
    __tablename__ = 'moisture_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    index_name = Column(String, nullable=False)
    image_name = Column(String, nullable=False)
    dry_percentage = Column(Float, nullable=False)
    normal_percentage = Column(Float, nullable=False)
    wet_percentage = Column(Float, nullable=False)
    per_pixel_accuracy = Column(Float, nullable=False)
    condition_summary = Column(String, nullable=False)

# Create tables
Base.metadata.create_all(engine)

def read_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode in ['RGB', 'RGBA']:
                img = img.convert('L')  # Convert to grayscale
            image_data = np.array(img)
            return image_data
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def normalize_image_data(image_data, min_value=None, max_value=None):
    if min_value is None or max_value is None:
        min_value, max_value = np.min(image_data), np.max(image_data)
    return (image_data - min_value) / (max_value - min_value)

def process_image(image_data, index_name, image_name):
    image_data = normalize_image_data(image_data)
    low_threshold = np.percentile(image_data, 10)
    high_threshold = np.percentile(image_data, 90)

    wet_conditions = image_data < low_threshold
    normal_conditions = (image_data >= low_threshold) & (image_data < np.median(image_data))
    dry_conditions = image_data >= np.median(image_data)

    classified_conditions = np.zeros(image_data.shape, dtype=int)
    classified_conditions[dry_conditions] = 0
    classified_conditions[normal_conditions] = 1
    classified_conditions[wet_conditions] = 2

    ground_truth_mask = np.random.choice([0, 1, 2], size=image_data.shape)  # Replace with actual ground truth
    flat_true_labels = ground_truth_mask.flatten()
    flat_pred_labels = classified_conditions.flatten()
    per_pixel_accuracy = accuracy_score(flat_true_labels, flat_pred_labels)

    dry_percentage = np.round(100 * np.sum(flat_pred_labels == 0) / len(flat_pred_labels), 2)
    normal_percentage = np.round(100 * np.sum(flat_pred_labels == 1) / len(flat_pred_labels), 2)
    wet_percentage = np.round(100 * np.sum(flat_pred_labels == 2) / len(flat_pred_labels), 2)

    # New classification logic
    if wet_percentage < 9.5 and dry_percentage > 51 and normal_percentage < 38:
        condition_summary = "Dry moisture conditions"
    elif wet_percentage > normal_percentage and wet_percentage > dry_percentage:
        condition_summary = "Wet moisture conditions"
    elif normal_percentage > wet_percentage and normal_percentage > dry_percentage:
        condition_summary = "Normal moisture conditions"
    else:
        condition_summary = "Mixed moisture conditions"

    # Save results to the database
    session = Session()
    try:
        result = MoistureResult(
            index_name=index_name,
            image_name=image_name,
            dry_percentage=dry_percentage,
            normal_percentage=normal_percentage,
            wet_percentage=wet_percentage,
            per_pixel_accuracy=per_pixel_accuracy,
            condition_summary=condition_summary
        )
        session.add(result)
        session.commit()
        print(f"Saved results for {image_name} in {index_name}")

        # Save results to a text file for Streamlit access
        with open(os.path.join(output_folder, f"{image_name}_results.txt"), 'w') as f:
            f.write(f"Index: {index_name}\n")
            f.write(f"Image Name: {image_name}\n")
            f.write(f"Dry Percentage: {dry_percentage}\n")
            f.write(f"Normal Percentage: {normal_percentage}\n")
            f.write(f"Wet Percentage: {wet_percentage}\n")
            f.write(f"Per Pixel Accuracy: {per_pixel_accuracy}\n")
            f.write(f"Condition Summary: {condition_summary}\n")

        # Save processed image as a colored output image
        colored_image = create_colored_output_image(classified_conditions)
        colored_image.save(os.path.join(output_folder, f"{image_name}_colored.png"))

    except Exception as e:
        session.rollback()
        print(f"Error saving results for {image_name} in {index_name}: {e}")
    finally:
        session.close()

def create_colored_output_image(classified_conditions):
    """Creates a colored output image based on classified moisture conditions."""
    color_map = {
        0: [255, 0, 0],  # Red for dry
        1: [0, 255, 0],  # Green for normal
        2: [0, 0, 255]   # Blue for wet
    }
    
    # Create a blank color image
    colored_image = np.zeros((*classified_conditions.shape, 3), dtype=np.uint8)

    # Apply the color map based on the classified conditions
    for condition, color in color_map.items():
        colored_image[classified_conditions == condition] = color

    return Image.fromarray(colored_image)

# Process each index
for index_name, folder_path in folder_paths.items():
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.png'):
            image_path = os.path.join(folder_path, image_file)
            print(f"Processing image: {image_path}")
            image_data = read_image(image_path)

            if image_data is not None:
                process_image(image_data, index_name, image_file)
            else:
                print(f"Failed to process image: {image_path}")
