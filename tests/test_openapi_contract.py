from pathlib import Path

import yaml

from app.main import app


def operations(schema):
    methods = {"get", "post", "patch", "delete", "put", "head", "options", "trace"}
    return {
        (path, method): set(operation["responses"]) - {"422"}
        for path, path_item in schema["paths"].items()
        for method, operation in path_item.items()
        if method in methods
    }


def test_documented_routes_and_response_codes_match_swagger():
    contract = Path(__file__).resolve().parents[1] / "docs/openapi.yaml"
    documented = operations(yaml.safe_load(contract.read_text(encoding="utf-8")))
    assert operations(app.openapi()) == documented
