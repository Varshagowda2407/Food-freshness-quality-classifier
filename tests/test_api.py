import io
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint_with_synthetic_image():
    # Create a 100x100 RGB dummy image in memory
    img = Image.new("RGB", (100, 100), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "confidence" in data
    assert data["status"] in ["Fresh", "Okay", "Avoid"]
