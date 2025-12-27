# find_unused.py
import ast
from pathlib import Path
from collections import defaultdict

path = Path("synthgen_mod.py")
tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))

defs = defaultdict(set)   # {"<global>": {...}, "RendererV3": {...}}
calls = defaultdict(set)  # имена вызываемых функций/методов

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.class_stack = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        owner = self.class_stack[-1] if self.class_stack else "<global>"
        defs[owner].add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # foo(...)
        if isinstance(node.func, ast.Name):
            calls["<any>"].add(node.func.id)
        # obj.foo(...)
        elif isinstance(node.func, ast.Attribute):
            calls["<any>"].add(node.func.attr)
        self.generic_visit(node)

Visitor().visit(tree)

called = calls["<any>"]

# не считаем магические методы (их часто вызывает Python сам)
ignore = {"__init__", "__enter__", "__exit__", "__iter__", "__next__"}

for owner, names in defs.items():
    unused = sorted([n for n in names if n not in called and n not in ignore])
    if unused:
        print(f"\n[{owner}] возможно не используется ({len(unused)}):")
        for n in unused:
            print("  -", n)
