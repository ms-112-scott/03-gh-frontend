import pathspec
from pathlib import Path
from typing import List


SEPARATOR: str = "-" * 92

CUSTOM_IGNORES: List[str] = [
    "__pycache__/",
    "*.pyc",
    ".venv/",
    "venv/",
    "env/",
    "Archive/",
    ".vscode/",
    "scripts/export_code_base.py",
    "*.json",
]


def get_ignore_spec(root_dir: Path) -> pathspec.PathSpec:
    ignore_lines: List[str] = list(CUSTOM_IGNORES)
    gitignore_path = root_dir / ".gitignore"

    if gitignore_path.exists():
        with gitignore_path.open("r", encoding="utf-8") as f:
            ignore_lines.extend(f.readlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)


def generate_folder_tree(
    dir_path: Path, spec: pathspec.PathSpec, root_dir: Path, prefix: str = ""
) -> str:
    tree_str = ""

    if prefix == "":
        tree_str += f"{dir_path.name}/\n"

    items = list(dir_path.iterdir())
    valid_items = []

    for item in items:
        if item.name == ".git":
            continue

        rel_path = item.relative_to(root_dir).as_posix()
        if item.is_dir():
            rel_path += "/"

        if not spec.match_file(rel_path):
            valid_items.append(item)

    valid_items.sort(key=lambda x: (x.is_file(), x.name))

    for i, item in enumerate(valid_items):
        is_last = i == len(valid_items) - 1
        connector = "`-- " if is_last else "|-- "

        tree_str += f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"

        if item.is_dir():
            extension = "    " if is_last else "|   "
            tree_str += generate_folder_tree(item, spec, root_dir, prefix + extension)

    return tree_str


def find_python_files(root_dir: Path, spec: pathspec.PathSpec) -> List[Path]:
    valid_files: List[Path] = []

    for file_path in root_dir.rglob("*.py"):
        rel_path = file_path.relative_to(root_dir).as_posix()
        if ".git" not in file_path.parts and not spec.match_file(rel_path):
            valid_files.append(file_path)

    return sorted(valid_files)


def export_codebase(root_dir: Path, output_file: Path) -> None:
    spec = get_ignore_spec(root_dir)
    py_files = find_python_files(root_dir, spec)
    tree_output = generate_folder_tree(root_dir, spec, root_dir)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("code folder structure...\n")
        f.write(SEPARATOR + "\n")
        f.write(tree_output)
        f.write("\n\n")

        f.write("code base\n")
        for file_path in py_files:
            rel_path: Path = file_path.relative_to(root_dir)

            f.write(SEPARATOR + "\n")
            f.write(str(rel_path) + "\n")

            try:
                code: str = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = file_path.read_text(encoding="latin-1")

            f.write(code)
            f.write("\n")

    print(f"Exported folder tree and {len(py_files)} python files -> {output_file}")


def main() -> None:
    root_dir: Path = Path(__file__).resolve().parents[1]
    output_file: Path = root_dir / "03-gh-frontend_code_base.txt"
    export_codebase(root_dir, output_file)


if __name__ == "__main__":
    main()
