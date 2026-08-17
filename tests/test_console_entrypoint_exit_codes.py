import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_SCRIPTS = (
    "src/isaac_so_arm101/scripts/list_envs.py",
    "src/isaac_so_arm101/scripts/zero_agent.py",
    "src/isaac_so_arm101/scripts/random_agent.py",
    "src/isaac_so_arm101/scripts/rsl_rl/train.py",
    "src/isaac_so_arm101/scripts/rsl_rl/play.py",
)


@pytest.mark.parametrize("relative_path", ENTRYPOINT_SCRIPTS)
def test_console_main_returns_integer_zero(relative_path: str) -> None:
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )

    assert not main_node.decorator_list
    assert isinstance(main_node.returns, ast.Name)
    assert main_node.returns.id == "int"

    final_statement = main_node.body[-1]
    assert isinstance(final_statement, ast.Return)
    assert isinstance(final_statement.value, ast.Constant)
    assert final_statement.value.value == 0
