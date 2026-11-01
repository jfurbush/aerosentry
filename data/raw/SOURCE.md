# FD001 data source

NASA CMAPSS Turbofan Engine Degradation Simulation, subset FD001 (simulated data, not real engines).
Original: NASA Prognostics Data Repository. A. Saxena, K. Goebel, D. Simon, N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM 2008.

These copies were downloaded from the public mirror `hankroark/Turbofan-Engine-Degradation`
on GitHub (folder `CMAPSSData`), because NASA's site could not be reached from the build
environment. They were checked against the documented FD001 shape (see
`tests/integration/test_real_fd001.py`), but not byte-compared with NASA's original download.
To verify, download the original archive from NASA and compare SHA-256 values.

| File | Rows | SHA-256 |
|------|------|---------|
| train_FD001.txt | 20631 | 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8 |
| test_FD001.txt | 13096 | 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851 |
| RUL_FD001.txt | 100 | a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca |
