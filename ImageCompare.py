from PIL import Image
import imagehash

# 載入圖片（請換成你自己的圖片路徑）
image1 = Image.open('image1.jpg')
image2 = Image.open('image2.jpg')

# 計算感知哈希
hash1 = imagehash.phash(image1)
hash2 = imagehash.phash(image2)

# 比較哈希距離
similarity_score = 1 - (hash1 - hash2) / len(hash1.hash) ** 2

print(f"Similarity: {similarity_score:.2f} between 1 to 0.")
