from functools import partial
import json
import ast
import contextlib
import io
from typing import Dict, Any
import os
import base64
import traceback
import hashlib

import tensorgrad
import sympy
from tensorgrad import functions
from tensorgrad.imgtools import save_steps

import boto3
from botocore.exceptions import ClientError

print(os.environ)

# Use the environment variable for the DynamoDB table name, with a default value.
DYNAMODB_TABLE = os.environ["DYNAMODB_TABLE"]

# Set up the DynamoDB resource and table.
dynamodb = boto3.resource("dynamodb")
cache_table = dynamodb.Table(DYNAMODB_TABLE)


def get_code_hash(code: str) -> str:
    """Generate a SHA256 hash for the given code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def get_cached_result(code: str):
    """Retrieve cached result from DynamoDB if available."""
    code_hash = get_code_hash(code)
    try:
        response = cache_table.get_item(Key={"code_hash": code_hash})
        if "Item" in response:
            return json.loads(response["Item"]["result"])
    except ClientError as e:
        print("DynamoDB get_item error:", e.response["Error"]["Message"])
    return None


def cache_result(code: str, result: dict):
    """Store the result in DynamoDB."""
    code_hash = get_code_hash(code)
    try:
        cache_table.put_item(
            Item={
                "code_hash": code_hash,
                "result": json.dumps(result)
                # Optionally, add a TTL attribute here if needed.
            }
        )
    except ClientError as e:
        print("DynamoDB put_item error:", e.response["Error"]["Message"])


class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes AST to detect potentially unsafe operations"""
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

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.blacklist:
                self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in self.blacklist:
            self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.blacklist:
            self.has_unsafe_ops = True
        self.generic_visit(node)


def safe_execute(code: str) -> Dict[str, Any]:
    """
    Safely execute user code with restrictions.
    Pre-generates any necessary files and captures stdout/stderr.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = {"success": False, "output": "", "error": "", "result": None}
    output_path = "/tmp/steps.png"

    try:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result["error"] = str(e)
            return result

        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        if analyzer.has_unsafe_ops:
            result["error"] = "Code contains unsafe operations"
            return result

        # Create a restricted globals dictionary
        restricted_globals = {
            "tg": tensorgrad,
            "F": functions,
            "sp": sympy,
            "save_steps": partial(save_steps, output_path=output_path),
        }

        local_namespace = {}
        # Execute code with stdout/stderr capture
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, restricted_globals, local_namespace)

        # Read the generated image (if any)
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                image_data = f.read()
            result["image"] = base64.b64encode(image_data).decode("utf-8")
            os.remove(output_path)
        else:
            result["image"] = None

        result["success"] = True
        result["output"] = stdout.getvalue()
        result["error"] = stderr.getvalue()

    except Exception as e:
        result["error"] = str(e)
        result["stacktrace"] = traceback.format_exc()

    return result


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function with DynamoDB caching.
    """
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "https://tensorcookbook.com",
        "Access-Control-Allow-Headers": "content-type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Cache-Control": "max-age=86400, public",
        "Vary": "Origin"
    }

    try:
        print(event)
        # For API Gateway, the payload is in event['body']
        if isinstance(event, dict) and "body" in event:
            try:
                payload = json.loads(event["body"])
                code = payload.get("code")
            except json.JSONDecodeError:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "Invalid JSON payload"}),
                }
        else:
            # Direct Lambda invocation
            code = event.get("code")

        if not code:
            return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "No code provided"})}

        # Check if a cached result exists
        cached_result = get_cached_result(code)
        if cached_result:
            print("Returning cached result")
            return {"statusCode": 200, "headers": headers, "body": json.dumps(cached_result)}

        # Execute code safely if no cache is found
        result = safe_execute(code)

        # Cache the result if execution was successful
        if result.get("success"):
            cache_result(code, result)

        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}

    except Exception as e:
        return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}

