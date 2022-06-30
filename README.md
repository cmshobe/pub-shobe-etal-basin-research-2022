# pub-shobe-etal-basin-research-2022
Code to accompany "Inverting passive margin stratigraphy for marine sediment transport dynamics over geologic time" by Charles M. Shobe et al. (2022; *Basin Research*). Please cite the paper if you use any of this material; this is how scientists get credit for their efforts to make their code [FAIR](https://www.go-fair.org/fair-principles/)!

See the preprint here: https://doi.org/10.31223/X5WS72

## Quick guide to repository contents

- This repository contains all code and data required to reproduce all analyses in the paper linked above.
- The [paper figures](https://github.com/cmshobe/pub-shobe-etal-basin-research-2022/tree/main/paper_figures) folder simply contains jupyter notebooks that reproduce figures 3-10 and S1-S4.
- The [marine](https://github.com/cmshobe/pub-shobe-etal-basin-research-2022/tree/main/marine) folder contains all data and modeling scripts, as well as `marine_environment.yaml` which will build the appropriate computing environment. Note that while all analyses can be conducted on a laptop/desktop machine, it is wise to use an HPC for inversion exercises if possible.
- The [marine](https://github.com/cmshobe/pub-shobe-etal-basin-research-2022/tree/main/marine) folder also contains all model and script output. For example, all results of all inversions, as well as all best fit model runs, are archived, as are files produced by the preprocessing (`prepro.py`) scripts. Some of these are sizable and are stored using Github large file storage.
