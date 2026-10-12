# Sources and dependencies

- The original 2048 game is by Gabriele Cirulli. This project implements the
  standard game rules independently; no upstream game source was found in the
  curated source set. The replay interface is the existing user-authored local
  interface, retained with its historical model identity.
- NumPy and PyTorch are installed dependencies, not vendored source or wheels.
  Their installed distributions include their own licenses and third-party
  notices; those terms remain applicable. NumPy's distribution contains multiple
  licensed components, so it should not be reduced to a blanket license for this
  repository.
- pytest and setuptools are test/build dependencies. GitHub's official checkout
  and setup-python actions are referenced by the CPU CI workflow.
- Source code hashes and curation notes are in docs/provenance.json. Operational
  deployment files and private infrastructure metadata are intentionally omitted.

No third-party license file existed in the supplied core/UI source trees, and no
new blanket license has been assigned to the original project contents.
