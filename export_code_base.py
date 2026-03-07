from pathlib import Path
from typing import List


SEPARATOR: str = "-" * 92


def find_python_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all .py files under root_dir.

    Parameters
    ----------
    root_dir : Path
        Root directory to search.

    Returns
    -------
    List[Path]
        List of python file paths.
    """
    return sorted(root_dir.rglob("*.py"))


def export_codebase(root_dir: Path, output_file: Path) -> None:
    """
    Export all Python files into a single text file.

    Each file is preceded by its relative path and separated
    by a long separator line.

    Parameters
    ----------
    root_dir : Path
        Root project directory.
    output_file : Path
        Output text file path.
    """

    py_files: List[Path] = find_python_files(root_dir)

    with output_file.open("w", encoding="utf-8") as f:

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

    print(f"Exported {len(py_files)} python files -> {output_file}")


def main() -> None:
    """
    Entry point.
    """

    root_dir: Path = Path("src")  # current directory
    output_file: Path = Path("code_base.txt")

    export_codebase(root_dir, output_file)


if __name__ == "__main__":
    main()
