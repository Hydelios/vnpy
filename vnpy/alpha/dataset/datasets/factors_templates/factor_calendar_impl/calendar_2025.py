from .catalog import CalendarFactorCatalog


def definitions():
    return CalendarFactorCatalog().build_year(2025)
