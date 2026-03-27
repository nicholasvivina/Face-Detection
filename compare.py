from deepface import DeepFace
from itertools import combinations

# ── Change these paths to your image paths ──
images = {
    "Image 1": r"C:\projects\deepface\face_test\img1.jpeg",
    "Image 2": r"C:\projects\deepface\face_test\img2.jpeg",
    "Image 3": r"C:\projects\deepface\face_test\img6.jpeg",
    "Image 4": r"C:\projects\deepface\face_test\img7.jpeg",
}

models = ["VGG-Face", "ArcFace", "Facenet512"]

print("\n=============================")
print("  FACE COMPARISON RESULTS")
print("=============================")

def compare(label, a, b):
    print(f"\n🔍 {label}")
    votes = 0
    for model in models:
        try:
            result = DeepFace.verify(
                img1_path=a,
                img2_path=b,
                model_name=model,
                detector_backend="retinaface",
                enforce_detection=True
            )
            same = result["verified"]
            distance = round(result["distance"], 4)
            threshold = round(result["threshold"], 4)
            if same:
                votes += 1
            print(f"   [{model}] {'✅ YES' if same else '❌ NO'} | Distance: {distance} | Threshold: {threshold}")
        except Exception as e:
            print(f"   [{model}] ⚠️ Error: {e}")

    print(f"\n   🗳️  Verdict: {'✅ SAME PERSON' if votes >= 2 else '❌ DIFFERENT PERSON'} ({votes}/3 models agree)")
    print("-" * 55)

# ── Auto generate all pairs from 4 images ──
image_pairs = list(combinations(images.keys(), 2))

for pair in image_pairs:
    label = f"{pair[0]} vs {pair[1]}"
    compare(label, images[pair[0]], images[pair[1]])

print("\n=============================")
print("  ALL COMPARISONS DONE ✅")
print("=============================\n")