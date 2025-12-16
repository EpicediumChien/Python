import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F # <-- Added import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import math

# Constants
CHAR_SET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(CHAR_SET)}
INDEX_TO_CHAR = {idx: ch for ch, idx in CHAR_TO_INDEX.items()}
CHAR_LEN = 4  # characters per CAPTCHA
IMAGE_PATH = r"C:\Git\Python\CAPCHA_SAMPLE" # <--- IMPORTANT: Make sure this path is correct

# --- NEW: Helper class to pad images to a square ---
class PadToSquare:
    """
    Pads a PIL image to a square shape by adding black padding to the shorter side.
    This preserves the aspect ratio of the content.
    """
    def __init__(self, fill=0): # 0 for black padding
        self.fill = fill

    def __call__(self, img):
        # Get image dimensions
        width, height = img.size
        
        # Find the longer side
        max_side = max(width, height)
        
        # Calculate the padding needed for width and height
        pad_width = max_side - width
        pad_height = max_side - height
        
        # (left, top, right, bottom)
        # We distribute padding to two sides to keep the image centered
        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
        
        return F.pad(img, padding, self.fill, 'constant')

# 1. Dataset class (No changes needed here)
class CaptchaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label_str = os.path.basename(path).split('-')[1].split('.')[0].upper()
        assert len(label_str) == CHAR_LEN

        img = Image.open(path).convert("L")  # grayscale

        width, height = img.size
        char_width = width // CHAR_LEN

        chars = []
        for i in range(CHAR_LEN):
            char_img = img.crop((i * char_width, 0, (i + 1) * char_width, height))
            if self.transform:
                char_img = self.transform(char_img)
            chars.append(char_img)

        images = torch.stack(chars)
        labels = torch.tensor([CHAR_TO_INDEX[c] for c in label_str])
        return images, labels

# 2. Vision Transformer model (No changes needed in the class itself)
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=1,
        num_classes=26,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    ):
        super().__init__()
        # --- CRITICAL: embed_dim must be divisible by num_heads ---
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
            
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * (patch_size ** 2)

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def _patchify(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, -1, c * self.patch_size * self.patch_size)
        return x

    def forward(self, x):
        x = self._patchify(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]
        return self.head(cls_output)


# 3. Load and split data
all_files = [
    os.path.join(IMAGE_PATH, f)
    for f in os.listdir(IMAGE_PATH)
    if f.endswith('.png') and '-' in f
]

if not all_files:
    raise FileNotFoundError(f"No images found in '{IMAGE_PATH}'. Please check the path and image filenames (e.g., 'sample-ABCD.png').")

train_paths, val_paths = train_test_split(all_files, test_size=0.1, random_state=42)

# --- NEW: Define target image and patch size ---
TARGET_IMG_SIZE = 64  # Pad and resize all characters to a 64x64 square
PATCH_SIZE = 8        # 64 is divisible by 8. This creates 8x8=64 patches.

# --- UPDATED: Transform pipeline with padding ---
transform = transforms.Compose([
    PadToSquare(fill=0),                                     # 1. Pad to a square, preserving aspect ratio
    transforms.Resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE)),   # 2. Resize to the target dimensions
    transforms.ToTensor(),                                   # 3. Convert to a tensor
])

train_dataset = CaptchaDataset(train_paths, transform)
val_dataset = CaptchaDataset(val_paths, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 4. Create 4 separate models (1 per character position)
# --- UPDATED: Instantiate model with new image and patch sizes ---
models = [VisionTransformer(
    img_size=TARGET_IMG_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=len(CHAR_SET),
    embed_dim=128,  # Good choice: 128 is divisible by 4
    depth=4,
    num_heads=4
) for _ in range(CHAR_LEN)]
criterions = [nn.CrossEntropyLoss() for _ in range(CHAR_LEN)]
optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) for model in models]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
for model in models:
    model.to(device)

# 5. Training loop (No changes needed here)
print("Starting training...")
for epoch in range(10):
    for model in models:
        model.train()
        
    for images_batch, labels_batch in train_loader:
        for i in range(CHAR_LEN):
            images = images_batch[:, i, :, :, :].to(device)
            labels = labels_batch[:, i].to(device)

            optimizers[i].zero_grad()
            outputs = models[i](images)
            loss = criterions[i](outputs, labels)
            loss.backward()
            optimizers[i].step()

    print(f"Epoch {epoch + 1} complete")

print("Training complete.")

# 6. (Optional) Save models
for i in range(CHAR_LEN):
    torch.save(models[i].state_dict(), f"captcha_char_transformer_model_{i}.pth")

# 7. Evaluation: Accuracy & MSE (No changes needed here)
print("\nEvaluating on validation set...")
for model in models:
    model.eval()

all_true = [[] for _ in range(CHAR_LEN)]
all_pred = [[] for _ in range(CHAR_LEN)]

with torch.no_grad():
    for images_batch, labels_batch in val_loader:
        for i in range(CHAR_LEN):
            images = images_batch[:, i, :, :, :].to(device)
            labels = labels_batch[:, i].to(device)

            outputs = models[i](images)
            preds = outputs.argmax(dim=1)

            all_true[i].extend(labels.cpu().numpy())
            all_pred[i].extend(preds.cpu().numpy())

total_acc, total_mse = 0, 0
for i in range(CHAR_LEN):
    acc = accuracy_score(all_true[i], all_pred[i])
    mse = mean_squared_error(all_true[i], all_pred[i])
    print(f"Char #{i+1} - Accuracy: {acc:.4f}, MSE: {mse:.4f}")
    total_acc += acc
    total_mse += mse

print(f"\nAverage Accuracy: {total_acc / CHAR_LEN:.4f}")
print(f"Average MSE: {total_mse / CHAR_LEN:.4f}")