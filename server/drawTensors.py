# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum
import json
import uuid
from datetime import datetime
import traceback
import os
import hashlib
import base64
import ast
import contextlib
import io
from typing import Optional
import tempfile

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

import tensorgrad
import sympy
from tensorgrad import functions
from tensorgrad.imgtools import save_steps
from functools import partial

app = FastAPI()

# ----------------- Pydantic Models -----------------


# Request Payload Model
class CodePayload(BaseModel):
    code: str


# Response model for code execution results
class ExecutionResult(BaseModel):
    success: bool
    output: str = ""
    error: str = ""
    image: Optional[str] = None
    stacktrace: Optional[str] = None


# Response model for snippet creation
class SnippetCreationResponse(BaseModel):
    snippet_id: str
    message: str = "Snippet created"


# Response model for snippet retrieval
class Snippet(BaseModel):
    snippet_id: str
    code: str
    created_at: str
    author_id: str


# ----------------- DynamoDB Setup -----------------


# Don't try to authenticate if not running as a Lambda function
if "DYNAMODB_CACHE_TABLE" in os.environ and "DYNAMODB_SNIPPET_TABLE" in os.environ:

    def check_aws_credentials() -> bool:
        """
        Try to use the STS service to check if AWS credentials are available.
        """
        try:
            client = boto3.client("sts")
            called_id = client.get_caller_identity()
            print("AWS Account ID:", called_id["Account"])
            return True
        except Exception:
            return False

    if check_aws_credentials():
        # Use DynamoDB if credentials are available
        dynamodb = boto3.resource("dynamodb")
        print(os.environ)
        cache_table = dynamodb.Table(os.environ["DYNAMODB_CACHE_TABLE"])
        snippet_table = dynamodb.Table(os.environ["DYNAMODB_SNIPPET_TABLE"])
    else:
        # Fallback: Use an in-memory "database"
        print("No AWS credentials available, using in-memory database.")

        class InMemoryTable:
            def __init__(self, key_name="id"):
                self.data = {}
                self.key_name = key_name

            def get_item(self, Key: dict) -> dict:
                key = Key[self.key_name]
                result = self.data.get(key)
                if result is None:
                    return {}
                return {"Item": self.data[key]}

            def put_item(self, Item: dict):
                key = Item[self.key_name]
                self.data[key] = Item

        cache_table = InMemoryTable(key_name="code_hash")
        snippet_table = InMemoryTable(key_name="snippet_id")

# ----------------- Helper Functions for Caching -----------------


def get_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def get_cached_result(code: str):
    code_hash = get_code_hash(code)
    try:
        response = cache_table.get_item(Key={"code_hash": code_hash})
        if "Item" in response:
            return json.loads(response["Item"]["result"])
    except ClientError as e:
        print("DynamoDB get_item error:", e.response["Error"]["Message"])
    except NoCredentialsError:
        print("No AWS credentials found")
    return None


def cache_result(code: str, result: dict):
    code_hash = get_code_hash(code)
    try:
        cache_table.put_item(
            Item={"code_hash": code_hash, "result": json.dumps(result)}
        )
    except ClientError as e:
        print("DynamoDB put_item error:", e.response["Error"]["Message"])
    except NoCredentialsError:
        print("No AWS credentials found")


# ----------------- Helper Functions for Snippet Sharing -----------------


def create_snippet(code: str, author_id: str = None) -> str:
    snippet_id = str(uuid.uuid4())
    try:
        snippet_table.put_item(
            Item={
                "snippet_id": snippet_id,
                "code": code,
                "created_at": datetime.utcnow().isoformat(),
                "author_id": author_id or "anonymous",
            }
        )
    except ClientError as e:
        print("DynamoDB put_item error (snippet):", e.response["Error"]["Message"])
        raise e
    except NoCredentialsError as e:
        print("No AWS credentials found", e)
        # We just return the snippet_id even if the save failed, helpful for testing
    return snippet_id


def get_snippet(snippet_id: str) -> dict:
    try:
        response = snippet_table.get_item(Key={"snippet_id": snippet_id})
        return response.get("Item")
    except ClientError as e:
        print("DynamoDB get_item error (snippet):", e.response["Error"]["Message"])
        return None
    except NoCredentialsError as e:
        print("No AWS credentials found", e)
        return None


# ----------------- Code Execution Safety Functions -----------------


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.has_unsafe_ops = False
        self.blacklist = {
            "os",
            "subprocess",
            "sys",
            "builtins",
            "eval",
            "exec",
            "globals",
            "locals",
            "compile",
            "__import__",
            "open",
            "importlib",
        }

    def get_full_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self.get_full_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def is_blacklisted(self, name: str) -> bool:
        # Split the name by dots and check each part.
        return any(part in self.blacklist for part in name.split("."))

    def visit_Import(self, node):
        for alias in node.names:
            # alias.name might be something like "os" or "os.path"
            if self.is_blacklisted(alias.name):
                self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and self.is_blacklisted(node.module):
            self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check both direct function calls and attribute calls (e.g., os.system)
        if isinstance(node.func, (ast.Name, ast.Attribute)):
            full_name = self.get_full_name(node.func)
            if self.is_blacklisted(full_name):
                self.has_unsafe_ops = True
        self.generic_visit(node)


def safe_execute(code: str) -> ExecutionResult:
    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(success=False, error=str(e))

        # Analyze for unsafe operations
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        if analyzer.has_unsafe_ops:
            return ExecutionResult(
                success=False,
                error="Code contains unsafe operations",
            )

        # Create a temporary file for image output.
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            # Set up a restricted execution environment, passing the temporary file's path
            restricted_globals = {
                "tg": tensorgrad,
                "F": functions,
                "sp": sympy,
                "save_steps": partial(save_steps, output_path=tmp.name),
            }
            local_namespace = {}

            # Execute the user code while capturing output and errors
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, restricted_globals, local_namespace)

            # Check if an image was generated
            image = None
            if os.path.exists(tmp.name):
                with open(tmp.name, "rb") as f:
                    image_data = f.read()
                # If the image is empty, we leave the image field as None
                if image_data:
                    base64data = base64.b64encode(image_data).decode('utf-8')
                    image = f"data:image/png;base64,{base64data}"

        return ExecutionResult(
            success=True,
            output=stdout.getvalue(),
            error=stderr.getvalue(),
            image=image,
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            stacktrace=traceback.format_exc()
        )

# ----------------- FastAPI Endpoints -----------------


@app.post("/execute", response_model=ExecutionResult)
async def execute_code(payload: CodePayload):
    code = payload.code
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")

    # Check for a cached result.
    cached = get_cached_result(code)
    if cached:
        return ExecutionResult(**cached)

    # Execute the code.
    result = safe_execute(code)

    # Cache the result if execution was successful.
    if result.success:
        cache_result(code, result.dict())

    return result


@app.post("/snippets", response_model=SnippetCreationResponse)
async def post_snippet(payload: CodePayload):
    code = payload.code
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    snippet_id = create_snippet(code)
    return SnippetCreationResponse(snippet_id=snippet_id)


@app.get("/snippets/{snippet_id}", response_model=Snippet)
async def fetch_snippet(snippet_id: str):
    snippet = get_snippet(snippet_id)
    if snippet is None:
        raise HTTPException(status_code=404, detail=f"Snippet {snippet_id} not found")
    return Snippet(**snippet)


# ----------------- Lambda Handler -----------------

handler = Mangum(app)
