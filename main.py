from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import replicate
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replicate client setup
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

@app.post("/generate/text-to-image/")
async def text_to_image(prompt: str = Form(...)):
    """Generate an image from text."""
    model = "stability-ai/stable-diffusion"
    version = "latest"
    try:
        output = replicate_client.run(
            model=model,
            version=version,
            input={"prompt": prompt}
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate/image-to-image/")
async def image_to_image(file: UploadFile, prompt: str = Form(...)):
    """Generate an image from another image."""
    model = "stability-ai/stable-diffusion-img2img"
    version = "latest"
    try:
        image_bytes = await file.read()
        output = replicate_client.run(
            model=model,
            version=version,
            input={"image": image_bytes, "prompt": prompt}
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate/inpainting/")
async def inpainting(file: UploadFile, mask: UploadFile, prompt: str = Form(...)):
    """Edit specific parts of an image based on a mask."""
    model = "stability-ai/stable-diffusion-inpainting"
    version = "latest"
    try:
        image_bytes = await file.read()
        mask_bytes = await mask.read()
        output = replicate_client.run(
            model=model,
            version=version,
            input={"image": image_bytes, "mask": mask_bytes, "prompt": prompt}
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}
