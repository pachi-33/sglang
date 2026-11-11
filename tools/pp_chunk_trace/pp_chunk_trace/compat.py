"""Check only the execution boundaries used by the patch, without importing SGLang."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

MANIFEST = Path(__file__).with_name("compatibility.json")


def fingerprint(source):
    tree = ast.parse(textwrap.dedent(source))
    # Python 3.12 added an empty type_params field to ordinary functions/classes.
    # Ignore it so a Mac 3.9 check agrees with the server's Python 3.12 check.
    for node in ast.walk(tree):
        if "type_params" in node._fields and not getattr(node, "type_params", None):
            node._fields = tuple(field for field in node._fields if field != "type_params")
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def verify_runtime(scheduler_module):
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest["methods"]:
        owner = getattr(scheduler_module, item["class"])
        method = getattr(owner, item["method"])
        if fingerprint(inspect.getsource(method)) != item["sha256"]:
            raise RuntimeError(f"Unsupported SGLang boundary: {item['class']}.{item['method']}; run python -m pp_chunk_trace.compat /path/to/sglang")
    return manifest["reference_commit"]


def check(repository):
    root = Path(repository).resolve()
    manifest = json.loads(MANIFEST.read_text())
    results = []
    for item in manifest["methods"]:
        path = root / item["path"]
        source = path.read_text()
        tree = ast.parse(source)
        found = None
        for owner in tree.body:
            if isinstance(owner, ast.ClassDef) and owner.name == item["class"]:
                for node in owner.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == item["method"]:
                        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        found = fingerprint("\n".join(source.splitlines()[first - 1:node.end_lineno]))
        results.append(dict(method=item["class"] + "." + item["method"], compatible=found == item["sha256"]))
    return dict(repository=str(root), reference_commit=manifest["reference_commit"],
                compatible=all(x["compatible"] for x in results), methods=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    args = parser.parse_args(argv)
    try:
        result = check(args.repository)
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["compatible"]:
        parser.exit(1, "Boundary implementation differs; review/adapt the patch before enabling.\n")


if __name__ == "__main__":
    main()
