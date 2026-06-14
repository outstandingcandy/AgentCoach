"""Vendored chat agent for AgentCore Runtime.

Mirror of the chat path from ``goalinsight/web/`` and
``goalinsight/highlights/_context.py``, copied flat so importing it
does NOT trigger the heavy ``goalinsight`` package __init__ (which
loads torch/cv2/wandb). Keep these files in lockstep with their
upstream counterparts when changing tool schemas or chat behaviour.
"""
