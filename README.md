# A waist at the binary collision: two-letter syzygy orbits of the strong-force three-body problem

Oleksii Stavriianov (ORCID [0009-0002-8504-3626](https://orcid.org/0009-0002-8504-3626))

The paper is in [`paper/main.pdf`](paper/main.pdf) (source `paper/main.tex`).

For the planar three-body problem with potential U = sum m_i m_j / r_ij^2, zero energy and zero angular momentum, the
solutions are the geodesics of Montgomery's Jacobi-Maupertuis metric on the shape sphere minus the three binary
collisions. The paper computes the leading curvature at a binary collision, K = kappa_2 rho^2 with kappa_2 in closed
form, and shows that when kappa_2 > 0, i.e. m_k^2 (m_i + m_j)^3 < m_i^2 m_j^2 M, the end has a waist and the
two-letter syzygy class (ij) is realised by a periodic orbit (periodic up to rotation); for masses (1, 1, m) the
threshold is m* = (1 + sqrt 65)/16. It describes the orbits, their birth from the cylinder at the threshold, and
reduces the converse (no waist, no orbit) to three conditions, the first proved by interval arithmetic, the other two
checked numerically.

## Repository

- `paper/`: the paper (LaTeX source, PDF, and the data of its figures in `paper/fig/`). It builds with any LaTeX
  distribution with pgfplots (for example `tectonic -X compile main.tex` in `paper/`).
- `verify/`: the verification supplement. Independent checkers (numpy, sympy, mpmath; no code shared with `tools/`),
  the certificates (initial conditions of the nine orbits, the chains of convex loops), and the saved output of every
  checker. See [`verify/README.md`](verify/README.md) for what each one checks and how long it runs. Every checker ends
  with ALL PASS or SOME FAILED. The searches, computations and scripts were carried out with the assistance of a large
  language model; the checkers were written independently of the research code.
- `tools/` and `words/tools/`: the research code behind the paper, as it was used (the reversible orbit search, the
  curve-shortening flow that produced the chains, the chain and Fermi-normal checks of the research stage, the map of
  the mass triangle, the birth-law constants, the stability and figure scripts). These scripts write into `words/out/`.
  `tools/words_alpha.py` is included for helper functions used by `tools/words_waist.py`; its own command-line entry
  needs a data file that is not part of this release.

## Quick check

    pip install numpy scipy sympy mpmath
    python verify/verify_threshold.py      # seconds
    python verify/verify_I_interval.py     # about a minute
    python verify/verify_orbits.py         # some minutes

## Licence

Code (`verify/`, `tools/`, `words/`): MIT, see [`LICENSE`](LICENSE). Paper text and figures (`paper/`): CC BY 4.0, see
[`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md).

## Citation

See [`CITATION.cff`](CITATION.cff); the Zenodo record of each release carries its DOI.
