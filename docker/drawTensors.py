from functools import partial
import json
import ast
import contextlib
import io
from typing import Dict, Any
import os
import base64
import traceback

import tensorgrad
import sympy
from tensorgrad import functions
from tensorgrad.imgtools import save_steps


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
        }

    def visit_Import(self, node):
        """Check for blacklisted imports"""
        for alias in node.names:
            if alias.name in self.blacklist:
                self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check for blacklisted from imports"""
        if node.module in self.blacklist:
            self.has_unsafe_ops = True
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check for calls to blacklisted functions"""
        if isinstance(node.func, ast.Name) and node.func.id in self.blacklist:
            self.has_unsafe_ops = True
        self.generic_visit(node)


def validate_code(code: str) -> bool:
    """
    Validate user code for safety
    Returns True if code is safe, False otherwise
    """
    try:
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        return not analyzer.has_unsafe_ops
    except SyntaxError:
        return False


def safe_execute(code: str) -> Dict[str, Any]:
    """
    Safely execute user code with restrictions
    """
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = {"success": False, "output": "", "error": "", "result": None}
    output_path = "/tmp/steps.png"

    try:
        # Validate code
        if not validate_code(code):
            result["error"] = "Code contains unsafe operations"
            return result

        # Create a restricted globals dictionary
        restricted_globals = {
            # '__builtins__': {
            #     name: getattr(__builtins__, name)
            #     for name in ['len', 'range', 'int', 'float', 'str', 'list', 'dict']
            # },
            "tg": tensorgrad,
            "F": functions,
            "sp": sympy,
            "save_steps": partial(save_steps, output_path=output_path),
        }

        local_namespace = {}
        # Execute code with stdout/stderr capture
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, restricted_globals, local_namespace)

        # Read the generated image
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                image_data = f.read()
            result["image"] = base64.b64encode(image_data).decode("utf-8")
            os.remove(output_path)  # Clean up
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
    AWS Lambda handler function
    """
    # Common headers for all responses
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "https://tensorcookbook.com",  # Allow CORS
        "Access-Control-Allow-Headers": "content-type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Cache-Control": "max-age=86400, public",  # 24 hours
        "Vary": "Origin"  # Important when using specific CORS origins
    }

    try:
        print(event)
        # For API Gateway requests, the payload is in event['body']
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

        # Execute code safely
        result = safe_execute(code)

        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}

    except Exception as e:
        return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}

