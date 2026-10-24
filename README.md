# ACDM Framework — Documentation Index

This directory contains the architectural and operational documentation for the Adaptive Containment Decision Manager framework.

## Read in this order

If you are new to the framework, the recommended reading order is:

1. **`scoring/ACDM_Scoring_Specification_v0.1.md`** — The mathematical foundation. What the framework computes and why. Start here for the technical core.

2. **`bluestrike/BlueStrike_v0.2_Adversarial_Robustness.md`** — The self-threat model. How the framework defends against attacks targeting the framework itself.

3. **`simulation/ACDM_Operational_Simulation_v0.1.md`** — The narrative walkthrough. A Tier-0 admin compromise scenario, end-to-end, with arithmetic decisions and timing.

4. **`pilot/ACDM_Pilot_Architecture_v0.1.md`** — The production engineering reference. How to actually build this for a real organization.

5. **`legal/ACDM_Legal_Compliance_Mapping_v0.1.md`** — The regulatory compatibility analysis. EU and MENA frameworks, with explicit identification of where legal counsel review is required.

6. **`visuals/`** — Diagrams and visual assets. Rendered as PNG/SVG; sources in `visuals/source/` (when present).

## Document conventions

- All documents are versioned (`_v0.1`, `_v0.2`, etc.)
- Cross-references between documents use relative links
- Status fields indicate the maturity level: draft, reviewed, ready for review
- The disclaimer at the top of each document is binding for any reader: this is reference material, not production system documentation, and the limitations sections are part of the substance

## License

The documents in this directory are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE.md` in this directory for details.

The simulator code in the top-level `/simulator/` directory is licensed separately under Apache License 2.0.

## Contributing

See `../CONTRIBUTING.md` for guidance on contributions. Note specifically the "adversarial-honesty rule": contributions that add capability claims must also document the limitations.

## Open work

This is `v0.1` of most documents. Known gaps include:

- A dedicated "Limitations & Counterarguments" document (currently the limitations are distributed across the documents, but a consolidated treatment is planned).
- Sector-specific compliance overlays for financial services, healthcare, and energy.
- A bayesian likelihood model calibration guide.
- Multi-language translations.

See `../CONTRIBUTING.md` for how to engage with these.
