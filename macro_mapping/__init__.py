# -*- coding: utf-8 -*-
"""
Behavioral Macro Mapping of LLM Decision Making

Builds a reasoning graph showing the model's step-by-step thought process
from raw input features to final CHOICE output. Uses counterfactual
perturbation to extract implicit feature exchange rates and decision boundaries.

Pipeline (10 steps):
  1. select_samples.py        — Pick 100 pairs (60 correct, 40 incorrect)
  2. extract_cot.py           — Collect and parse chain-of-thought text
  3. setup_perturbations.py   — Define perturbation targets and increments
  4. run_perturbations.py     — Run perturbation queries, record flip points
  5. map_reasoning_stages.py  — Identify recurring reasoning stages from CoT
  6. compute_exchange_rates.py — Derive feature exchange rates from flip points
  7. build_reasoning_graph.py — Construct the visual reasoning graph
  8. decision_boundary.py     — Add decision boundary sub-graph
  9. validate_regression.py   — Compare with logistic regression baseline
 10. produce_final_graph.py   — Combine into final annotated deliverable
"""
