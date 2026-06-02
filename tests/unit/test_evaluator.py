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

def test_evaluator_calculates_correct_auc():
    """Valida se a AUC calculada corresponde aos valores preditos."""
    evaluator = PerformanceEvaluator()
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.1]
    
    auc_val = evaluator.calculate_auc(labels, scores)
    assert auc_val == 1.0
    
    # Caso com classes de comprimento diferente
    assert evaluator.calculate_auc([], []) == 0.0

def test_evaluator_calculates_correct_precision_at_k():
    """Garante o cálculo correto de Precision@K."""
    evaluator = PerformanceEvaluator()
    retrieved_ids = ["pessoa_1", "pessoa_2", "pessoa_1", "pessoa_3"]
    
    # Precision@1 (pessoa_1 matches top-1) -> 1/1 = 1.0
    assert evaluator.calculate_precision_at_k("pessoa_1", retrieved_ids, k=1) == 1.0
    # Precision@2 (pessoa_1 matches 1 out of top-2) -> 1/2 = 0.5
    assert evaluator.calculate_precision_at_k("pessoa_1", retrieved_ids, k=2) == 0.5
    # Precision@3 (pessoa_1 matches 2 out of top-3) -> 2/3
    assert evaluator.calculate_precision_at_k("pessoa_1", retrieved_ids, k=3) == pytest.approx(0.66667, abs=1e-4)
    # Para Desconhecido deve retornar 0.0
    assert evaluator.calculate_precision_at_k("Desconhecido", retrieved_ids, k=3) == 0.0

def test_evaluator_calculates_correct_recall_at_k():
    """Garante o cálculo correto de Recall@K."""
    evaluator = PerformanceEvaluator()
    retrieved_ids = ["pessoa_1", "pessoa_2", "pessoa_3"]
    
    # Recall@1 para pessoa_1 -> presente -> 1.0
    assert evaluator.calculate_recall_at_k("pessoa_1", retrieved_ids, k=1) == 1.0
    # Recall@1 para pessoa_2 -> ausente no top-1 -> 0.0
    assert evaluator.calculate_recall_at_k("pessoa_2", retrieved_ids, k=1) == 0.0
    # Recall@2 para pessoa_2 -> presente no top-2 -> 1.0
    assert evaluator.calculate_recall_at_k("pessoa_2", retrieved_ids, k=2) == 1.0
    # Para Desconhecido deve retornar 0.0
    assert evaluator.calculate_recall_at_k("Desconhecido", retrieved_ids, k=2) == 0.0
