import pytest
from src.core.base import FaceRecognizer

def test_cannot_instantiate_abstract_base_class():
    """Garante que a classe base abstrata FaceRecognizer não possa ser instanciada diretamente."""
    with pytest.raises(TypeError):
        FaceRecognizer()

def test_concrete_subclass_must_implement_all_methods():
    """Garante que subclasses que não implementem todos os métodos abstratos levantem TypeError."""
    class IncompleteRecognizer(FaceRecognizer):
        pass

    with pytest.raises(TypeError):
        IncompleteRecognizer()
