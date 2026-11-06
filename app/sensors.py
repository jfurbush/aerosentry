"""Sensor names and units for the NASA CMAPSS telemetry, and the smoothing used on the chart.

The names and units are from Table 2, "C-MAPSS outputs to measure system response", in
A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM 2008. The 21 rows are the sensors in the data files, in order.
The dataset is simulated.
"""

from dataclasses import dataclass

NO_UNIT = "no unit"
MOVING_AVERAGE_WINDOW = 10


@dataclass(frozen=True)
class Sensor:
    column: str
    symbol: str
    description: str
    unit: str

    @property
    def number(self) -> int:
        return int(self.column[1:])

    @property
    def label(self) -> str:
        """Text for the sensor picker, for example "Sensor 11: Ps30 (Static pressure ..., psia)"."""
        return f"Sensor {self.number}: {self.symbol} ({self.description}, {self.unit})"

    @property
    def axis_title(self) -> str:
        return f"{self.symbol}: {self.description} ({self.unit})"

    @property
    def hover_unit(self) -> str:
        """Unit to append to a value in a chart tooltip. Empty when the quantity has no unit."""
        return "" if self.unit == NO_UNIT else f" {self.unit}"


_TABLE: tuple[tuple[str, str, str], ...] = (
    ("T2", "Total temperature at fan inlet", "°R"),
    ("T24", "Total temperature at LPC outlet", "°R"),
    ("T30", "Total temperature at HPC outlet", "°R"),
    ("T50", "Total temperature at LPT outlet", "°R"),
    ("P2", "Pressure at fan inlet", "psia"),
    ("P15", "Total pressure in bypass-duct", "psia"),
    ("P30", "Total pressure at HPC outlet", "psia"),
    ("Nf", "Physical fan speed", "rpm"),
    ("Nc", "Physical core speed", "rpm"),
    ("epr", "Engine pressure ratio (P50/P2)", NO_UNIT),
    ("Ps30", "Static pressure at HPC outlet", "psia"),
    ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    ("NRf", "Corrected fan speed", "rpm"),
    ("NRc", "Corrected core speed", "rpm"),
    ("BPR", "Bypass Ratio", NO_UNIT),
    ("farB", "Burner fuel-air ratio", NO_UNIT),
    ("htBleed", "Bleed Enthalpy", NO_UNIT),
    ("Nf_dmd", "Demanded fan speed", "rpm"),
    ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    ("W31", "HPT coolant bleed", "lbm/s"),
    ("W32", "LPT coolant bleed", "lbm/s"),
)

SENSORS: tuple[Sensor, ...] = tuple(
    Sensor(column=f"s{i}", symbol=symbol, description=description, unit=unit)
    for i, (symbol, description, unit) in enumerate(_TABLE, start=1)
)

# The sensor shown when the page opens. In FD001 it is one of the clearest trends with wear.
DEFAULT_SENSOR = "s11"


def moving_average(values: list[float], window: int = MOVING_AVERAGE_WINDOW) -> list[float | None]:
    """Trailing average of the last `window` values.

    The first window - 1 entries are None, so the line starts once a full window exists and
    never shows an average of fewer readings than it claims.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    out: list[float | None] = [None] * min(window - 1, len(values))
    total = sum(values[: window - 1])
    for i in range(window - 1, len(values)):
        total += values[i]
        out.append(round(total / window, 4))
        total -= values[i - window + 1]
    return out
