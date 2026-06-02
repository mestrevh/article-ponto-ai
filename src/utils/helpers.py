# Utilitários gerais e gerenciamento de caminhos do projeto
def get_project_root() -> str:
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
