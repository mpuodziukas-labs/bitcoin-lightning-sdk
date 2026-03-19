"""
Lightning Network pathfinding.

Provides:
  - Graph: weighted directed graph of Lightning channels
  - dijkstra(graph, src, dst): minimum-fee path using Dijkstra's algorithm
  - probability_weighted_path: path that maximises success probability
  - mpp_split: split a payment into at most max_parts parts (MPP)
  - fee_calculate: compute routing fee for a single hop
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """A directed Lightning channel edge."""

    source: str
    destination: str
    channel_id: str
    base_fee_msat: int        # fixed fee in millisatoshis
    fee_rate_ppm: int         # proportional fee in parts-per-million
    capacity_msat: int        # channel capacity in millisatoshis
    min_htlc_msat: int = 1_000
    max_htlc_msat: int = 0    # 0 = use capacity
    cltv_delta: int = 40

    @property
    def effective_max_htlc_msat(self) -> int:
        return self.max_htlc_msat if self.max_htlc_msat > 0 else self.capacity_msat


class Graph:
    """
    Directed weighted graph of Lightning Network channels.

    Nodes are identified by string (typically pubkey hex or alias).
    Multiple parallel channels between the same pair of nodes are supported.
    """

    def __init__(self) -> None:
        self._edges: dict[str, list[Channel]] = {}
        self._nodes: set[str] = set()

    def add_channel(self, channel: Channel) -> None:
        """Add a directed channel edge."""
        self._nodes.add(channel.source)
        self._nodes.add(channel.destination)
        self._edges.setdefault(channel.source, []).append(channel)

    def add_bidirectional_channel(
        self,
        node_a: str,
        node_b: str,
        channel_id: str,
        base_fee_msat: int = 1_000,
        fee_rate_ppm: int = 100,
        capacity_msat: int = 1_000_000_000,
    ) -> None:
        """Convenience: add both directions of a channel."""
        self.add_channel(Channel(node_a, node_b, channel_id, base_fee_msat, fee_rate_ppm, capacity_msat))
        self.add_channel(Channel(node_b, node_a, channel_id + "_r", base_fee_msat, fee_rate_ppm, capacity_msat))

    def neighbors(self, node: str) -> list[Channel]:
        return self._edges.get(node, [])

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

def fee_calculate(amount_msat: int, base_fee_msat: int, fee_rate_ppm: int) -> int:
    """
    Calculate the routing fee for forwarding *amount_msat* through a channel.

    fee = base_fee_msat + ceil(amount_msat * fee_rate_ppm / 1_000_000)

    Parameters
    ----------
    amount_msat:  Amount being forwarded in millisatoshis.
    base_fee_msat: Fixed base fee in millisatoshis.
    fee_rate_ppm:  Proportional fee in parts-per-million.

    Returns
    -------
    Total fee in millisatoshis (integer, rounded up).
    """
    if amount_msat < 0:
        raise ValueError("amount_msat must be non-negative.")
    if base_fee_msat < 0:
        raise ValueError("base_fee_msat must be non-negative.")
    if fee_rate_ppm < 0:
        raise ValueError("fee_rate_ppm must be non-negative.")
    proportional = math.ceil(amount_msat * fee_rate_ppm / 1_000_000)
    return base_fee_msat + proportional


# ---------------------------------------------------------------------------
# Dijkstra (minimum fee)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PQEntry:
    cost: int
    node: str = field(compare=False)
    path: list[str] = field(compare=False)


def dijkstra(graph: Graph, src: str, dst: str, amount_msat: int = 1_000) -> Optional[list[str]]:
    """
    Find the minimum-fee path from *src* to *dst* for *amount_msat*.

    Parameters
    ----------
    graph:       The Lightning Network graph.
    src:         Source node identifier.
    dst:         Destination node identifier.
    amount_msat: Amount to route in millisatoshis.

    Returns
    -------
    List of node identifiers from src to dst (inclusive), or None if no path.
    """
    if src == dst:
        return [src]
    if src not in graph.nodes or dst not in graph.nodes:
        return None

    dist: dict[str, int] = {src: 0}
    pq: list[_PQEntry] = [_PQEntry(cost=0, node=src, path=[src])]

    while pq:
        entry = heapq.heappop(pq)
        current_cost = entry.cost
        current_node = entry.node
        current_path = entry.path

        if current_node == dst:
            return current_path

        if current_cost > dist.get(current_node, math.inf):
            continue

        for channel in graph.neighbors(current_node):
            neighbor = channel.destination
            if amount_msat < channel.min_htlc_msat:
                continue
            if amount_msat > channel.effective_max_htlc_msat:
                continue
            edge_fee = fee_calculate(amount_msat, channel.base_fee_msat, channel.fee_rate_ppm)
            new_cost = current_cost + edge_fee
            if new_cost < dist.get(neighbor, math.inf):
                dist[neighbor] = new_cost
                heapq.heappush(
                    pq,
                    _PQEntry(cost=new_cost, node=neighbor, path=current_path + [neighbor]),
                )

    return None


# ---------------------------------------------------------------------------
# Probability-weighted path
# ---------------------------------------------------------------------------

def probability_weighted_path(
    graph: Graph,
    src: str,
    dst: str,
    success_rates: dict[str, float],
    amount_msat: int = 1_000,
) -> Optional[list[str]]:
    """
    Find the path that maximises end-to-end payment success probability.

    Uses Dijkstra on the negative log of success probability
    (so minimising -log(p) = maximising log(p) = maximising p).

    Parameters
    ----------
    graph:         The Lightning Network graph.
    src:           Source node identifier.
    dst:           Destination node identifier.
    success_rates: Dict mapping channel_id → success probability [0, 1].
                   Missing channel_ids default to 0.5.
    amount_msat:   Amount to route in millisatoshis.

    Returns
    -------
    List of node identifiers representing the highest-probability path, or None.
    """
    if src == dst:
        return [src]
    if src not in graph.nodes or dst not in graph.nodes:
        return None

    _INF = float("inf")

    # neg_log_prob: lower is better (higher probability)
    neg_log_dist: dict[str, float] = {src: 0.0}

    @dataclass(order=True)
    class _PEntry:
        cost: float
        node: str = field(compare=False)
        path: list[str] = field(compare=False)

    pq2: list[_PEntry] = [_PEntry(cost=0.0, node=src, path=[src])]

    while pq2:
        entry2 = heapq.heappop(pq2)
        curr_cost = entry2.cost
        curr_node = entry2.node
        curr_path = entry2.path

        if curr_node == dst:
            return curr_path

        if curr_cost > neg_log_dist.get(curr_node, _INF):
            continue

        for channel in graph.neighbors(curr_node):
            neighbor = channel.destination
            if amount_msat < channel.min_htlc_msat:
                continue
            if amount_msat > channel.effective_max_htlc_msat:
                continue
            p = success_rates.get(channel.channel_id, 0.5)
            p = max(1e-9, min(1.0, p))  # clamp to valid range
            edge_cost = -math.log(p)
            new_cost = curr_cost + edge_cost
            if new_cost < neg_log_dist.get(neighbor, _INF):
                neg_log_dist[neighbor] = new_cost
                heapq.heappush(
                    pq2,
                    _PEntry(cost=new_cost, node=neighbor, path=curr_path + [neighbor]),
                )

    return None


# ---------------------------------------------------------------------------
# MPP split
# ---------------------------------------------------------------------------

def mpp_split(amount_msat: int, max_parts: int = 8, min_part_msat: int = 10_000) -> list[int]:
    """
    Split *amount_msat* into at most *max_parts* roughly-equal parts for
    Multi-Path Payments (MPP).

    Constraints:
    - Each part >= min_part_msat.
    - Parts sum exactly to amount_msat.
    - Number of parts <= max_parts.
    - Parts are as equal as possible.

    Parameters
    ----------
    amount_msat:   Total amount to split in millisatoshis.
    max_parts:     Maximum number of parts.
    min_part_msat: Minimum size of each part in millisatoshis.

    Returns
    -------
    List of part sizes in millisatoshis.

    Raises
    ------
    ValueError: if the constraints cannot be satisfied.
    """
    if amount_msat <= 0:
        raise ValueError("amount_msat must be positive.")
    if max_parts <= 0:
        raise ValueError("max_parts must be positive.")
    if min_part_msat <= 0:
        raise ValueError("min_part_msat must be positive.")

    # How many parts can we actually fit given the minimum?
    max_possible = amount_msat // min_part_msat
    if max_possible == 0:
        raise ValueError(
            f"amount_msat {amount_msat} is less than min_part_msat {min_part_msat}."
        )

    n_parts = min(max_parts, max_possible)

    base = amount_msat // n_parts
    remainder = amount_msat % n_parts

    parts = [base + (1 if i < remainder else 0) for i in range(n_parts)]
    assert sum(parts) == amount_msat
    return parts
