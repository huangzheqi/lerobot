import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "src/isaac_so_arm101/scripts/rsl_rl/train.py"
TRAIN_SOURCE = TRAIN_PATH.read_text(encoding="utf-8")
TRAIN_TREE = ast.parse(TRAIN_SOURCE)


def _function_source(name: str) -> str:
    node = next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    source = ast.get_source_segment(TRAIN_SOURCE, node)
    assert source is not None
    return source


def test_fast_exit_argument_is_opt_in_and_headless_only() -> None:
    argument_call = next(
        node
        for node in ast.walk(TRAIN_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--fast_exit"
    )
    argument_source = ast.get_source_segment(TRAIN_SOURCE, argument_call)

    assert argument_source is not None
    assert 'action="store_true"' in argument_source
    assert "default=False" in argument_source
    assert "if args_cli.fast_exit and not args_cli.headless:" in TRAIN_SOURCE
    assert 'parser.error("--fast_exit requires --headless.")' in TRAIN_SOURCE


def test_fast_exit_flushes_logs_before_hard_exit() -> None:
    helper_source = _function_source("_fast_exit_after_training")

    assert 'getattr(runner, "writer", None)' in helper_source
    assert 'getattr(writer, "flush", None)' in helper_source
    assert "sys.stdout.flush()" in helper_source
    assert "sys.stderr.flush()" in helper_source
    assert "os._exit(0)" in helper_source


def test_fast_exit_runs_after_learning_and_preserves_default_cleanup() -> None:
    run_source = _function_source("_run")
    learn_index = run_source.index("runner.learn(")
    condition_index = run_source.index("if args_cli.fast_exit:", learn_index)
    fast_exit_index = run_source.index("_fast_exit_after_training(runner)", condition_index)
    env_close_index = run_source.index("env.close()", fast_exit_index)

    assert learn_index < condition_index < fast_exit_index < env_close_index
    assert "simulation_app.close()" in TRAIN_SOURCE
