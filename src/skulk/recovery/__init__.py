"""Restart recovery: bring placed models and steward history back after a cold restart.

``record`` defines the small per-node record of what a cold-started cluster
should restore and derives it from a replicated view; ``recorder`` keeps the
record current on each node's disk. A master that starts with no prior
replicated view (a single-node reboot or a whole-cluster restart) reads it
back. See ``website/docs/architecture.md`` ("Restart recovery").
"""
