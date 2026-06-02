import os
from src.utils.helpers import get_project_root

def test_get_project_root():
    """Garante que a função get_project_root retorne o caminho correto do projeto contendo pyproject.toml."""
    root = get_project_root()
    assert os.path.exists(root)
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
