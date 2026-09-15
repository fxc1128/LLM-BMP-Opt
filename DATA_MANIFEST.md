# Public Example Assets

Large model assets are tracked with Git LFS.

- Youwuzhen SEIMS: official preparation inputs, 30 m MongoDB dump, scenario configuration, SEIMS source, and Windows binaries.
- Fountain Creek SWAT+: complete TxtInOut workspace including SWAT+.exe.
- Urban LID: EPA SWMM 5.2.4 Windows runtime and the published bioretention.inp LID model.

The Youwuzhen package excludes OUTPUT_*, SA_NSGA2_*, surrogate-training, and workspace directories because they are historical outputs rather than inputs required for a new run.

SEIMS source and binaries are GPL-3.0. The license copy is licenses/SEIMS-GPL-3.0.txt. Upstream source: https://github.com/lreis2415/SEIMS.

EPA SWMM 5.2.4 is public domain. bioretention.inp originates from the OpenWaterAnalytics SWMM non-regression test suite and is distributed under CC BY 4.0; its license copy is licenses/OWA-SWMM-NRTESTSUITE-CC-BY-4.0.txt.
