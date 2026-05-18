from PIL import Image

# Load sour
img = Image.open("target.jpg")

# Resize to 512x512
img_resized = img.resize((512, 512), Image.LANCZOS)

# Save
img_resized.save("target.jpg")

print("Saved resized image:", img_resized.size)