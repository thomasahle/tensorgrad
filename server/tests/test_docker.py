import sys
import subprocess
import time
import requests
import pytest
import json
import socket
from typing import Optional
from pydantic import BaseModel

from server.drawTensors import CodePayload, ExecutionResult, SnippetCreationResponse, Snippet

PORT = 9000  # Host port
LAMBDA_URL = f"http://localhost:{PORT}/2015-03-31/functions/function/invocations"


def wait_for_port(host: str, port: int, timeout: float = 10.0):
    """Wait for a network port to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((host, port))
            if result == 0:
                return True
        time.sleep(0.5)
    raise TimeoutError(f"Timeout waiting for {host}:{port} to become available.")


@pytest.fixture(scope="session")
def docker_container():
    """
    Build the Docker image once per session, run the container, yield the container ID,
    and then stop and remove the container after the session ends.
    """

    # Run the build command with live output.
    # We need to run from the tensorgrad root directory
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try buildx first, fall back to regular build
    build_cmd = ["docker", "buildx", "build"]
    
    # Check if buildx is available
    buildx_check = subprocess.run(
        ["docker", "buildx", "version"],
        capture_output=True,
        text=True
    )
    
    if buildx_check.returncode != 0:
        print("Note: docker buildx not available, using regular docker build")
        build_cmd = ["docker", "build"]

    # Build the Docker image
    build_proc = subprocess.run(
        build_cmd + [
            "-t",
            "tensorgrad",
            "-f",
            "server/Dockerfile",
            ".",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert build_proc.returncode == 0, f"Docker build failed:\n{build_proc.stderr}"

    # Run the Docker container in detached mode (mapping host port PORT to container port 8080).
    run_proc = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "lambda_local_test",
            "-p",
            f"{PORT}:8080",
            "tensorgrad",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert run_proc.returncode == 0, f"Docker run failed:\n{run_proc.stderr}"
    container_id = run_proc.stdout.strip()

    # Allow time for the container to start.
    wait_for_port("localhost", PORT, timeout=10)
    time.sleep(1)
    print("Docker container is running.")

    try:
        yield container_id

    finally:
        print("Cleaning up Docker container...")
        logs = subprocess.run(
            ["docker", "logs", container_id], capture_output=True, text=True
        )
        print("\nDocker container logs:\n", logs.stdout, logs.stderr)

        # Teardown: stop and remove the container.
        subprocess.run(
            ["docker", "stop", container_id], stdout=sys.stdout, stderr=sys.stderr
        )
        subprocess.run(
            ["docker", "rm", container_id], stdout=sys.stdout, stderr=sys.stderr
        )


def invoke_api(
    body: Optional[BaseModel] = None,
    path: str = "/",
    method: str = "GET",
    pathParameters: dict = None,
) -> requests.Response:
    """
    Helper function that creates a simulated API Gateway event and invokes the Lambda endpoint.
    """
    response = requests.post(
        LAMBDA_URL,
        json={
            "resource": path,
            "path": path,
            "httpMethod": method,
            "queryStringParameters": None,
            "pathParameters": pathParameters,
            "body": None if body is None else body.model_dump_json(),
            "isBase64Encoded": False,
            "requestContext": {
                "resourcePath": path,
                "httpMethod": method,
                "path": path,
            },
        },
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200, f"API failed: {response.text}"
    print(response)
    data = response.json()
    assert data["statusCode"] == 200, f"API failed: {data['body']}"
    print("reponse.json()=", data)
    return json.loads(data["body"])


def test_docker_image_execute_endpoint(docker_container):
    payload_obj = CodePayload(code="print('Hello from Docker')")
    response = invoke_api(body=payload_obj, path="/execute", method="POST")
    result = ExecutionResult.model_validate(response)
    assert result.success is True, f"Execution did not succeed: {result}"


def test_docker_snippet_endpoints(docker_container):
    # ---- Snippet Creation ----
    payload_obj = CodePayload(code="print('Hello, snippet!')")
    response_body = invoke_api(body=payload_obj, path="/snippets", method="POST")
    snippet_creation = SnippetCreationResponse.model_validate(response_body)
    assert snippet_creation.snippet_id is not None
    snippet_id = snippet_creation.snippet_id

    # ---- Snippet Retrieval ----
    response_get = invoke_api(
        path=f"/snippets/{snippet_id}",
        method="GET",
        pathParameters={"snippet_id": snippet_id},
    )
    snippet_obj = Snippet.model_validate(response_get)
    assert snippet_obj.snippet_id == snippet_id
    assert snippet_obj.code == payload_obj.code
    assert snippet_obj.created_at is not None
    assert snippet_obj.author_id is not None
