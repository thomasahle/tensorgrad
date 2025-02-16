import os
import json
import base64
import pytest
import uuid
import textwrap
from datetime import datetime

from fastapi.testclient import TestClient
from moto import mock_aws  # Use the generic AWS mock

# Set the environment variables before importing your app.
os.environ["DYNAMODB_CACHE_TABLE"] = "TensorgradCache"
os.environ["DYNAMODB_SNIPPET_TABLE"] = "CodeSnippets"

# Now import the app and helper functions from your FastAPI application.
from drawTensors import app, safe_execute, create_snippet, get_snippet, ExecutionResult

# Create a TestClient for FastAPI
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_dynamodb():
    """
    Use Moto to mock AWS and create the necessary DynamoDB tables for testing.
    """
    with mock_aws():
        import boto3  # Import boto3 within the Moto context

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

        # Create the CodeCache table
        try:
            dynamodb.create_table(
                TableName="CodeCache",
                KeySchema=[{"AttributeName": "code_hash", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "code_hash", "AttributeType": "S"}
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )
        except Exception:
            pass

        # Create the CodeSnippets table
        try:
            dynamodb.create_table(
                TableName="CodeSnippets",
                KeySchema=[{"AttributeName": "snippet_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "snippet_id", "AttributeType": "S"}
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )
        except Exception:
            pass

        # Yield to run tests inside the Moto context.
        yield


# -------- Tests for safe_execute --------


def test_safe_execute_success():
    code = "print('Hello, world!')"
    result: ExecutionResult = safe_execute(code)
    assert result.success is True
    assert "Hello, world!" in result.output
    # For this code, there should be no image, error, or stacktrace.
    assert result.image is None
    assert result.error == ""
    assert result.stacktrace is None


def test_safe_execute_syntax_error():
    code = "print('Hello"  # unbalanced quote causes SyntaxError
    result: ExecutionResult = safe_execute(code)
    assert result.success is False
    assert "SyntaxError" in result.error or "unterminated" in result.error


def test_safe_execute_unsafe_code():
    # Attempt to import os should be blocked.
    code = "import os\nprint('Unsafe')"
    result: ExecutionResult = safe_execute(code)
    assert result.success is False
    assert "unsafe operations" in result.error


# -------- Tests for /execute endpoint --------


def test_execute_endpoint_success():
    payload = {"code": "print('Test Execute')"}
    response = client.post("/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Validate using the ExecutionResult schema fields.
    assert data["success"] is True
    assert "Test Execute" in data["output"]
    assert data["error"] == ""


def test_execute_good_code_with_image():
    code = textwrap.dedent(
        """
        i = sp.symbols('i');
        x = tg.Delta(i, 'i', 'j');
        y = x * 2;
        save_steps(y);
    """
    )
    response = client.post("/execute", json={"code": code})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "image" in data
    # parse image
    image_data = data["image"]
    assert image_data.startswith("data:image/png;base64,")


def test_execute_endpoint_invalid_payload():
    # Send an empty payload.
    response = client.post("/execute", json={"code": ""})
    assert response.status_code == 400
    data = response.json()
    assert "No code provided" in data["detail"]


# -------- Tests for Snippet Endpoints --------


def test_create_and_fetch_snippet():
    code = "print('Snippet Test')"
    # Create a snippet.
    response_post = client.post("/snippets", json={"code": code})
    assert response_post.status_code == 200
    post_data = response_post.json()
    assert "snippet_id" in post_data
    snippet_id = post_data["snippet_id"]

    # Retrieve the snippet.
    response_get = client.get(f"/snippets/{snippet_id}")
    assert response_get.status_code == 200
    get_data = response_get.json()
    # Verify that the snippet contains the correct code.
    assert get_data["snippet_id"] == snippet_id
    assert get_data["code"] == code
    # Verify created_at and author_id are present.
    assert "created_at" in get_data
    assert "author_id" in get_data


def test_fetch_nonexistent_snippet():
    # Attempt to fetch a snippet that doesn't exist.
    fake_snippet_id = str(uuid.uuid4())
    response_get = client.get(f"/snippets/{fake_snippet_id}")
    assert response_get.status_code == 404
    data = response_get.json()
    assert "not found" in data["detail"]
