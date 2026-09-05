"""Provide the subset of the old pyairports dataset required by Outlines."""

import airportsdata


def _load_airport_list():
    """Return stable legacy rows whose fourth item is a unique IATA code."""
    codes = {
        airport["iata"].upper()
        for airport in airportsdata.load().values()
        if airport.get("iata")
    }
    return [("", "", "", code) for code in sorted(codes)]


AIRPORT_LIST = _load_airport_list()
