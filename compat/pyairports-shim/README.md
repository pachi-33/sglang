# pyairports compatibility shim

Outlines 0.0.46 imports `AIRPORT_LIST` from `pyairports.airports`, but the
`pyairports` distribution currently available from PyPI does not provide that
module. This local package preserves the small legacy API that Outlines needs
and obtains the airport data from the maintained `airportsdata` package.

The fourth item in each `AIRPORT_LIST` row is the IATA code, matching the only
field read by Outlines 0.0.46.

Install the shim from the root of the SGLang worktree:

```bash
python -m pip uninstall -y pyairports
python -m pip install --no-build-isolation -e ./compat/pyairports-shim
```
