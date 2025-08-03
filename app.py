import gradio as gr
import os
import shutil
import uuid
from inference import run
from omegaconf import OmegaConf
import argparse
import glob
from PIL import Image

# Setup default values
CONFIG_PATH = "configs/stableviton.yaml"  # Make sure this exists
MODEL_LOAD_PATH = "checkpoints/stableviton.ckpt"  # Adjust to match your weight
SAVE_DIR = "samples"
TEST_DIR = "test"

# Ensure folder structure
def setup_test_dir():
    for subfolder in ["image", "cloth", "openpose-img", "openpose-json", "image-parse", "agnostic", "cloth-mask"]:
        path = os.path.join(TEST_DIR, subfolder)
        os.makedirs(path, exist_ok=True)

# Save uploaded images to expected folders
def save_inputs(person_img, cloth_img):
    uid = str(uuid.uuid4())[:8]
    person_path = os.path.join(TEST_DIR, "image", f"{uid}_person.jpg")
    cloth_path = os.path.join(TEST_DIR, "cloth", f"{uid}_cloth.jpg")
    person_img.save(person_path)
    cloth_img.save(cloth_path)
    return uid, person_path, cloth_path

# Run inference script
def generate(person_img, cloth_img):
    setup_test_dir()
    uid, person_path, cloth_path = save_inputs(person_img, cloth_img)

    args = argparse.Namespace(
        config_path=CONFIG_PATH,
        model_load_path=MODEL_LOAD_PATH,
        batch_size=1,
        data_root_dir=TEST_DIR,
        repaint=False,
        unpair=True,
        save_dir=SAVE_DIR,
        denoise_steps=70,
        img_H=512,
        img_W=384,
        eta=0.0
    )

    # Call the main function
    run(args)

    # Find the generated image
    output_pattern = os.path.join(SAVE_DIR, f"{uid}_person**{uid}_cloth.png")
    outputs = glob.glob(output_pattern)
    if outputs:
        return Image.open(outputs[0])
    else:
        return "Output image not found."

# Gradio Interface
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil", label="Person Image"),
        gr.Image(type="pil", label="Cloth Image")
    ],
    outputs=gr.Image(type="pil", label="Output"),
    title="StableVITON Try-On (Render.com)"
)

iface.launch(server_name="0.0.0.0", server_port=8080)
