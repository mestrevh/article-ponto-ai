from typing import Dict, Optional

class TrackVotingSystem:
    def __init__(self, min_votes: int = 1):
        self.min_votes = min_votes
        self.votes = {}  # {track_id: {identity: count}}

    def add_vote(self, track_id: int, identity: str) -> None:
        """Adiciona um voto de identidade para o track_id correspondente."""
        if track_id not in self.votes:
            self.votes[track_id] = {}
        self.votes[track_id][identity] = self.votes[track_id].get(identity, 0) + 1

    def get_votes(self, track_id: int) -> Dict[str, int]:
        """Retorna todos os votos acumulados de um determinado track_id."""
        return self.votes.get(track_id, {})

    def get_winner(self, track_id: int) -> Optional[str]:
        """Determina a identidade vencedora por maioria de votos, respeitando o mínimo de votos."""
        track_votes = self.get_votes(track_id)
        if not track_votes:
            return None
            
        total_votes = sum(track_votes.values())
        if total_votes < self.min_votes:
            return None
            
        # Retorna o que tem maior número de votos
        winner = max(track_votes, key=track_votes.get)
        return winner
