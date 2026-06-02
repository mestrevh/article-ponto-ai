import pytest
from src.core.voting import TrackVotingSystem

def test_voting_system_accumulates_votes():
    """Valida se os votos de identidade se acumulam por track_id."""
    voter = TrackVotingSystem()
    
    voter.add_vote(track_id=1, identity="user_a")
    voter.add_vote(track_id=1, identity="user_a")
    voter.add_vote(track_id=1, identity="user_b")
    
    assert voter.get_votes(track_id=1) == {"user_a": 2, "user_b": 1}

def test_voting_system_resolves_majority_winner():
    """Garante que a identidade mais votada vence o track."""
    voter = TrackVotingSystem()
    voter.add_vote(track_id=1, identity="user_a")
    voter.add_vote(track_id=1, identity="user_b")
    voter.add_vote(track_id=1, identity="user_b")
    
    assert voter.get_winner(track_id=1) == "user_b"

def test_voting_system_ignores_low_persistence_tracks():
    """Garante o descarte de tracks com pouca persistência (ex: menos de 3 votos)."""
    voter = TrackVotingSystem(min_votes=3)
    voter.add_vote(track_id=2, identity="user_a")
    voter.add_vote(track_id=2, identity="user_a")
    
    assert voter.get_winner(track_id=2) is None
