import pytest

from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_without_file(client):
    response = client.post("/predict")
    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "Файл не передан"}
