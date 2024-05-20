class Error(Exception):
    """Base class for exceptions."""

    pass


class XMLError(Error):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(Error):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(Error):
    """Exception raised for really really bad RNG."""

    pass
