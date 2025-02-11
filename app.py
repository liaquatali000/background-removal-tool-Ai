import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import os
from zipfile import ZipFile
import shutil
import glob

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Directory setup
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_images_from_data_dir():
    """Get all image files from the data directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(DATA_DIR, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(DATA_DIR, f'*{ext.upper()}')))
    return sorted(image_files)

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def process_bulk_images(files, progress=gr.Progress()):
    if not files:
        return None, "No files selected for processing.", [], "0/0"
    
    # Initialize results tracking
    results = []
    file_count = len(files)
    
    # Process images one by one
    for idx, file in enumerate(files, 1):
        try:
            filename = os.path.basename(file)
            progress(idx/file_count, f"Processing {filename} ({idx}/{file_count})")
            
            # Process image
            im = load_img(file, output_type="pil")
            im = im.convert("RGB")
            processed = process(im)
            
            # Save processed image to output directory
            output_filename = f"processed_{filename}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            processed.save(output_path)
            
            # Add to results
            results.append([
                filename,
                "✅ Completed",
                output_path
            ])
            
            # Update progress
            yield (
                None,  # zip file not ready yet
                f"Processing: {idx}/{file_count} images completed",
                results,
                f"{idx}/{file_count}"
            )
            
        except Exception as e:
            results.append([
                filename,
                f"❌ Error: {str(e)}",
                None
            ])
    
    # Create ZIP file with all successful outputs
    zip_path = os.path.join(OUTPUT_DIR, "processed_images.zip")
    with ZipFile(zip_path, 'w') as zip_file:
        for result in results:
            if result[2]:  # If there's a valid output path
                zip_file.write(result[2], os.path.basename(result[2]))
    
    successful = len([r for r in results if '✅' in r[1]])
    failed = len([r for r in results if '❌' in r[1]])
    final_status = f"Complete! Processed {file_count} images: {successful} successful, {failed} failed"
    
    yield zip_path, final_status, results, f"{file_count}/{file_count}"

def update_file_list():
    """Get list of available images in data directory"""
    files = get_images_from_data_dir()
    if not files:
        return [], f"No images found in {DATA_DIR} directory"
    return files, f"Found {len(files)} images in {DATA_DIR} directory"

# Create the Gradio interface
with gr.Blocks(title="Background Removal Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖼️ Background Removal Tool")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📁 Select Images")
            
            with gr.Row():
                input_images = gr.File(
                    label="Upload your own images",
                    file_count="multiple",
                    type="filepath",
                    file_types=["image"],
                    scale=3
                )
                load_dir_btn = gr.Button("📂 Load from Data Folder", scale=1)
            
            file_status = gr.Markdown("")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Actions")
            process_btn = gr.Button("▶️ Start Processing", variant="primary", size="lg")
            progress_status = gr.Markdown("0/0 images processed")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Processing Results")
            results_table = gr.Dataframe(
                headers=["Filename", "Status", "Output Path"],
                label="",
                interactive=False,
                value=[],
                wrap=True
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📦 Download")
            output_zip = gr.File(
                label="Download all processed images as ZIP",
                file_types=["zip"]
            )
            gr.Markdown(f"*Individual processed images can be found in the '{OUTPUT_DIR}' folder*")
    
    # Event handlers
    load_dir_btn.click(
        fn=update_file_list,
        outputs=[input_images, file_status]
    )
    
    process_btn.click(
        fn=process_bulk_images,
        inputs=[input_images],
        outputs=[output_zip, file_status, results_table, progress_status],
        show_progress=True
    )
    
    input_images.change(
        fn=lambda files: f"Selected {len(files)} images" if files else "No images selected",
        inputs=[input_images],
        outputs=[file_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)