# MSML640_Group_Project

Implement services/simulator_importer.py
Parse scrambles (text → CubeState).
Generate random scrambles.
Optionally read/export JSON cube states.
Deliver scan_cube(scramble: str = None) -> CubeState.
Provide clean inputs for solvers and UI.
Implement core/cube.py: Cube class, state representation, moves.
Implement core/notation.py: parse & format move sequences.
Implement validators (color counts, solvability checks).
Deliver the engine that makes solving possible.
Implement core/beginner_solver.py with 6 LBL stages.
Each stage returns moves + explanation.
Provide teaching-friendly solutions.
Ensure integration with Cube model.
Implement services/pipeline.py: glue importer + solver.
Provide CLI tool:
python -m services.pipeline --scramble "R U R' U'" --mode beginner
Add optional kociemba_solver.py for advanced mode.
Handle logging, fallbacks, metrics.
Sole focus: Quality assurance across all modules.
Responsibilities:
Create tests/ folder with pytest structure.
Write unit tests for:
Cube moves (B)
Scramble importer (A)
Beginner solver (C)
Pipeline (D)
Write baseline test suite: 100 scrambles → ensure solver completes with valid solution.
Maintain CI/CD config (.github/workflows/ci.yml) to run pytest + coverage.
Track bug reports, log reproducible steps, confirm fixes.
Measure performance: average solve length, time.
Deliver: Confidence that the system works under different inputs before release.

📌 Acceptance Gates (Owned by QA Person)

Baseline 1 (Week 3): Importer + Cube + simple solver stages tested with 10 scrambles.
Baseline 2 (Week 5): End-to-end pipeline tested with 50 scrambles.
Baseline 3 (Week 8): Full app tested with 100 scrambles; all pass.
✅ Benefits of This Setup

Builders (A–D) move fast without worrying about writing tests.
Tester (E) becomes the single source of truth for quality and reproducibility.
Team avoids duplicated test effort and has objective baselines before the due date.
