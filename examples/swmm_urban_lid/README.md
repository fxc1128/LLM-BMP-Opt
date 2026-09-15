# Urban EPA SWMM Bioretention Example

The real engine is external/swmm/runtime/runswmm.exe. The published North Bioretention LID model is data/swmm/bioretention/bioretention.inp.

Profile urban_bioretention_epa_swmm invokes the engine for every candidate. Decision 0 removes the LID usage and decision 1 enables the published bioretention control. Results are parsed from the generated SWMM report: total runoff volume, peak runoff, and LID cost.
