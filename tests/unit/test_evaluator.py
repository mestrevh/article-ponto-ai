import pytest
from src.metrics.evaluator import PerformanceEvaluator

def test_evaluator_calculates_correct_accuracy():
    """Valida se a acurácia reflete exatamente a proporção de acertos."""
    evaluator = PerformanceEvaluator()
    y_true = ["user_a", "user_b", "user_c"]
    y_pred = ["user_a", "user_b", "desconhecido"]
    
    accuracy = evaluator.calculate_accuracy(y_true, y_pred)
    assert accuracy == pytest.approx(0.66667, abs=1e-4)

def test_evaluator_tar_at_far_calculation():
    """Valida o cálculo de TAR @ FAR fixada em 1%."""
    evaluator = PerformanceEvaluator()
    scores = [0.9, 0.85, 0.8, 0.2, 0.1, 0.05]
    labels = [1, 1, 1, 0, 0, 0]
    
    tar = evaluator.calculate_tar_at_far(labels, scores, far_threshold=0.01)
    assert 0.0 <= tar <= 1.0
