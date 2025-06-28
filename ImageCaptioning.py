from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


image_path = "download.jpg"  
image = Image.open("download.jpg").convert('RGB')


inputs = processor(images=image, return_tensors="pt")

out = model.generate(**inputs, max_length=50)  
caption = processor.decode(out[0], skip_special_tokens=True)


print("🖼️ Generated Caption:", caption)
