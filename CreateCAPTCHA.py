import torch
import random, string
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random, string
## Final CAPTCHA Generator Code

# Step 1: Generate Random Text
def random_string(length=4, charset=None):    
    if charset is None:
        charset = random.choice([
            string.ascii_uppercase,  
            # string.digits,  
            # string.ascii_uppercase + string.digits  # Mix of letters & digits
        ])
    return ''.join(random.choice(charset) for _ in range(length))

# Step 2: Generate CAPTCHA with Rotated Characters
def rotated_twist_captcha(image, text, width=135, height=39):
    
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    x_offset = 10  # Start position
    char_y_base = height // 2 - 12  # Center text with small shift

    for char in text:
        # Create individual letter images
        char_img = Image.new('RGBA', (50, 50), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((10, 10), char, font=font, fill=(random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)))

        # Rotate randomly
        angle = random.randint(-25, 25)  
        char_img = char_img.rotate(angle, expand=True)

        # Define adjusted char_y
        #char_y = max(2, min(char_y_base + random.randint(-3, 3), height - 15))
        char_y = -10 # to avoid text cut-off

        # Paste rotated character
        image.paste(char_img, (x_offset, char_y), char_img)
        x_offset += 20  

    return image

def add_noise(image):    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(width * height // 50):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

def add_lines(image):    
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(5):
        # Pick one side to start from (left, right, top, bottom)
        side = random.choice(['top', 'bottom', 'left', 'right'])

        if side == 'top':
            x1, y1 = random.randint(0, width), -10
            x2, y2 = random.randint(0, width), height + 10
        elif side == 'bottom':
            x1, y1 = random.randint(0, width), height + 10
            x2, y2 = random.randint(0, width), -10
        elif side == 'left':
            x1, y1 = -10, random.randint(0, height)
            x2, y2 = width + 10, random.randint(0, height)
        else:  # right
            x1, y1 = width + 10, random.randint(0, height)
            x2, y2 = -10, random.randint(0, height)

        line_color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        line_width = 4
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)


def distorted_captcha(text, width=135, height=39):  
    image = Image.new('RGB', (width, height), color=(random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)))
    add_lines(image)  
    rotated_twist_captcha(image, text, width, height)
    add_noise(image)
    return image

# Generate and Display 5 CAPTCHAs in One Row
plt.figure(figsize=(20, 5))

# for i in range(100):
for i in range(275):
    text = random_string(length=4)
    image = distorted_captcha(text, width=135, height=39)

    # plt.subplot(1, 5, i + 1)
    # plt.imshow(image)
    # plt.title(f'CAPTCHA: {text}')
    # plt.axis('off')
    image.save(f'CAPCHA_SAMPLE\sample-{text}.png')  # Save all examples

# plt.tight_layout()
# plt.show()