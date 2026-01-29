"""
Extract all images from Jupyter notebooks and save them to an images/ folder.

This script walks through all .ipynb files in the workspace, extracts any 
embedded images (PNG, JPG, SVG) from cell outputs, and saves them with 
descriptive filenames based on the folder and notebook name.

Usage:
    python extract_notebook_images.py
"""

import json
import base64
import os
from pathlib import Path

def extract_images_from_notebook(notebook_path, output_dir):
    """Extract all images from a Jupyter notebook and save them."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get folder and notebook name for organizing images
    notebook_name = Path(notebook_path).stem
    parent_folder = Path(notebook_path).parent.name
    
    # Create subfolder for this topic
    topic_output_dir = output_dir / parent_folder
    topic_output_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    
    # Iterate through cells
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell['cell_type'] != 'code':
            continue
            
        # Check for outputs with images
        outputs = cell.get('outputs', [])
        for output_idx, output in enumerate(outputs):
            # Handle different output types
            image_data = None
            extension = None
            
            # Check for image data in various formats
            if 'data' in output:
                data = output['data']
                if 'image/png' in data:
                    image_data = data['image/png']
                    extension = 'png'
                elif 'image/jpeg' in data:
                    image_data = data['image/jpeg']
                    extension = 'jpg'
                elif 'image/svg+xml' in data:
                    image_data = data['image/svg+xml']
                    extension = 'svg'
            
            if image_data:
                # Create filename
                filename = f"{notebook_name}_cell{cell_idx+1}_img{output_idx+1}.{extension}"
                filepath = topic_output_dir / filename
                
                # Decode and save image
                if extension == 'svg':
                    # SVG is text-based
                    if isinstance(image_data, list):
                        image_data = ''.join(image_data)
                    with open(filepath, 'w', encoding='utf-8') as img_file:
                        img_file.write(image_data)
                else:
                    # PNG and JPG are base64 encoded
                    if isinstance(image_data, list):
                        image_data = ''.join(image_data)
                    # Remove any newlines
                    image_data = image_data.replace('\n', '')
                    with open(filepath, 'wb') as img_file:
                        img_file.write(base64.b64decode(image_data))
                
                image_count += 1
                print(f"  Saved: {filepath}")
    
    return image_count

def main():
    """Main function to extract images from all notebooks."""
    
    # Get the script's directory (workspace root)
    workspace_root = Path(__file__).parent
    
    # Create output directory for images
    output_dir = workspace_root / "images"
    output_dir.mkdir(exist_ok=True)
    
    print("Extracting images from notebooks...")
    print(f"Output directory: {output_dir}\n")
    
    total_images = 0
    total_notebooks = 0
    
    # Walk through all subdirectories looking for notebooks
    for notebook_path in workspace_root.rglob("*.ipynb"):
        # Skip checkpoints
        if '.ipynb_checkpoints' in str(notebook_path):
            continue
        
        print(f"Processing: {notebook_path.relative_to(workspace_root)}")
        
        try:
            count = extract_images_from_notebook(notebook_path, output_dir)
            total_images += count
            total_notebooks += 1
            
            if count == 0:
                print("  No images found")
            else:
                print(f"  Extracted {count} image(s)")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    print("-" * 60)
    print(f"Complete! Extracted {total_images} images from {total_notebooks} notebooks.")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()
