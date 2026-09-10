from PIL import Image
from pathlib import Path

INPUT_FOLDER = Path(r"D:\StateOfSiege Normandy\MANUAL NEW CARD IMAGES\printER-READY DECK")
OUTPUT_FOLDER = Path(r"D:\StateOfSiege Normandy\MANUAL NEW CARD IMAGES\printER-READY DECK\BLEED")

BLEED = 72

OUTPUT_FOLDER.mkdir(exist_ok=True)

for input_file in INPUT_FOLDER.glob("*.png"):
    img = Image.open(input_file)

    w, h = img.size
    new = Image.new(img.mode, (w + 2 * BLEED, h + 2 * BLEED))

    # Paste original image
    new.paste(img, (BLEED, BLEED))

    # Extend top edge
    top = img.crop((0, 0, w, 1)).resize((w, BLEED))
    new.paste(top, (BLEED, 0))

    # Extend bottom edge
    bottom = img.crop((0, h - 1, w, h)).resize((w, BLEED))
    new.paste(bottom, (BLEED, BLEED + h))

    # Extend left edge
    left = img.crop((0, 0, 1, h)).resize((BLEED, h))
    new.paste(left, (0, BLEED))

    # Extend right edge
    right = img.crop((w - 1, 0, w, h)).resize((BLEED, h))
    new.paste(right, (BLEED + w, BLEED))

    # Fill corners with corner pixels
    new.paste(Image.new(img.mode, (BLEED, BLEED), img.getpixel((0, 0))), (0, 0))
    new.paste(Image.new(img.mode, (BLEED, BLEED), img.getpixel((w - 1, 0))), (BLEED + w, 0))
    new.paste(Image.new(img.mode, (BLEED, BLEED), img.getpixel((0, h - 1))), (0, BLEED + h))
    new.paste(Image.new(img.mode, (BLEED, BLEED), img.getpixel((w - 1, h - 1))), (BLEED + w, BLEED + h))

    output_file = OUTPUT_FOLDER / input_file.name
    new.save(output_file)

print("Finished.")