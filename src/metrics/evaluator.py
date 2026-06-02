from typing import List
import numpy as np

class PerformanceEvaluator:
    def calculate_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """Calcula a taxa de acerto global (Accuracy)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        return float(correct) / len(y_true)

    def calculate_tar_at_far(self, labels: List[int], scores: List[float], far_threshold: float) -> float:
        """
        Calcula a taxa de aceitação verdadeira (TAR) para uma determinada
        taxa de falsa aceitação (FAR) estabelecida.
        """
        if not labels or not scores or len(labels) != len(scores):
            return 0.0
            
        labels_arr = np.array(labels)
        scores_arr = np.array(scores)
        
        # Ordena os scores dos impostores (labels == 0) para encontrar o limiar do threshold correspondente ao FAR
        impostor_scores = scores_arr[labels_arr == 0]
        if len(impostor_scores) == 0:
            return 1.0 # Sem impostores, aceitação total é trivial
            
        impostor_scores_sorted = np.sort(impostor_scores)
        # O limiar é o valor do percentil correspondente a (1 - far_threshold)
        idx = int(np.floor((1.0 - far_threshold) * len(impostor_scores_sorted)))
        idx = min(max(idx, 0), len(impostor_scores_sorted) - 1)
        threshold = impostor_scores_sorted[idx]
        
        # Calcula TAR: proporção de genuínos (labels == 1) com score >= threshold
        genuine_scores = scores_arr[labels_arr == 1]
        if len(genuine_scores) == 0:
            return 0.0
            
        true_accepts = np.sum(genuine_scores >= threshold)
        return float(true_accepts) / len(genuine_scores)

    def calculate_auc(self, y_true: List[int], y_scores: List[float]) -> float:
        """Calcula a Área sob a Curva ROC (AUC)."""
        if not y_true or not y_scores or len(y_true) != len(y_scores):
            return 0.0
        try:
            from sklearn.metrics import roc_curve, auc
            # Se todas as amostras forem da mesma classe, a AUC não é bem definida.
            if len(set(y_true)) < 2:
                return 1.0 if (1 in y_true and all(y_scores)) else 0.0
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            return float(auc(fpr, tpr))
        except Exception:
            return 0.0

    def calculate_precision_at_k(self, true_id: str, retrieved_ids: List[str], k: int) -> float:
        """Calcula a precisão dos K vizinhos mais próximos recuperados."""
        if not retrieved_ids or k <= 0:
            return 0.0
        if true_id == "Desconhecido":
            return 0.0
        top_k = retrieved_ids[:k]
        matches = sum(1 for rid in top_k if rid == true_id)
        return float(matches) / k

    def calculate_recall_at_k(self, true_id: str, retrieved_ids: List[str], k: int) -> float:
        """Calcula o recall dos K vizinhos mais próximos recuperados (1.0 ou 0.0)."""
        if not retrieved_ids or k <= 0:
            return 0.0
        if true_id == "Desconhecido":
            return 0.0
        top_k = retrieved_ids[:k]
        return 1.0 if true_id in top_k else 0.0
