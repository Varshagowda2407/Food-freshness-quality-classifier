from PIL import Image
from transformers import pipeline

class FreshnessClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )

    def predict(self, image: Image.Image) -> dict:
        results = self.classifier(image)
        top = results[0]
        label = top["label"].lower()

        # Map predictions to freshness categories
        if any(w in label for w in ["fresh", "ripe", "edible"]):
            status = "Fresh"
        elif any(w in label for w in ["rotten", "spoiled", "mold"]):
            status = "Avoid"
        else:
            status = "Okay"

        return {
            "status": status,
            "raw_label": top["label"],
            "confidence": f"{round(top['score'] * 100, 2)}%"
        }
