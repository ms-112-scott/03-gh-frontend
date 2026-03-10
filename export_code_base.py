import pathspec
from pathlib import Path
from typing import List


SEPARATOR: str = "-" * 92

# 在這裡定義你的「自訂忽略規則」(語法與 .gitignore 相同)
CUSTOM_IGNORES: List[str] = [
    "__pycache__/",
    "*.pyc",
    ".venv/",
    "venv/",
    "env/",
    "Archive/",
    ".vscode/",
    "export_code_base.py",
    "*.json",
]


def get_ignore_spec(root_dir: Path) -> pathspec.PathSpec:
    """
    解析 .gitignore 與自訂規則，回傳統一的 PathSpec 物件。
    """
    ignore_lines: List[str] = list(CUSTOM_IGNORES)
    gitignore_path = root_dir / ".gitignore"

    if gitignore_path.exists():
        with gitignore_path.open("r", encoding="utf-8") as f:
            ignore_lines.extend(f.readlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)


def generate_folder_tree(
    dir_path: Path, spec: pathspec.PathSpec, root_dir: Path, prefix: str = ""
) -> str:
    """
    遞迴產生資料夾結構樹狀圖字串，並自動略過被忽略的檔案與資料夾。
    """
    tree_str = ""

    # 處理最上層根目錄的名稱顯示
    if prefix == "":
        tree_str += f"{dir_path.name}/\n"

    items = list(dir_path.iterdir())
    valid_items = []

    for item in items:
        # 強制忽略 Git 內部資料夾
        if item.name == ".git":
            continue

        # 轉換為 POSIX 相對路徑以供 pathspec 比對
        rel_path = item.relative_to(root_dir).as_posix()

        # 關鍵：針對資料夾加上結尾斜線，確保 `__pycache__/` 這種針對資料夾的忽略規則能正確命中
        if item.is_dir():
            rel_path += "/"

        # 如果沒有被忽略，才加入有效清單
        if not spec.match_file(rel_path):
            valid_items.append(item)

    # 排序：資料夾排在前面，檔案排後面，然後依字母順序排列
    valid_items.sort(key=lambda x: (x.is_file(), x.name))

    for i, item in enumerate(valid_items):
        is_last = i == len(valid_items) - 1
        connector = "└─ " if is_last else "├─ "

        tree_str += f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"

        # 如果是資料夾，遞迴往下長出樹枝
        if item.is_dir():
            extension = "   " if is_last else "│  "
            tree_str += generate_folder_tree(item, spec, root_dir, prefix + extension)

    return tree_str


def find_python_files(root_dir: Path, spec: pathspec.PathSpec) -> List[Path]:
    """
    遞迴尋找所有符合條件的 .py 檔案。
    """
    valid_files: List[Path] = []

    for file_path in root_dir.rglob("*.py"):
        rel_path = file_path.relative_to(root_dir).as_posix()
        if ".git" not in file_path.parts and not spec.match_file(rel_path):
            valid_files.append(file_path)

    return sorted(valid_files)


def export_codebase(root_dir: Path, output_file: Path) -> None:
    """
    將資料夾結構與 Python 程式碼合併匯出至單一檔案。
    """
    # 1. 取得忽略規則
    spec = get_ignore_spec(root_dir)

    # 2. 獲取樹狀結構字串與程式碼清單
    py_files = find_python_files(root_dir, spec)
    tree_output = generate_folder_tree(root_dir, spec, root_dir)

    with output_file.open("w", encoding="utf-8") as f:
        # 寫入目錄樹狀圖
        f.write("code folder structure...\n")
        f.write(SEPARATOR + "\n")
        f.write(tree_output)
        f.write("\n\n")

        # 寫入 Codebase 主體
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
    """
    Entry point.
    """
    root_dir: Path = Path(
        "C:/Users/GAI/Desktop/NCA_workspace/03-gh-frontend"
    )  # current directory
    output_file: Path = Path("03-gh-frontend_code_base.txt")

    export_codebase(root_dir, output_file)


if __name__ == "__main__":
    main()
