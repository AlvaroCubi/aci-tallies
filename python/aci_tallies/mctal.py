"""MCNP MCTAL files: the F tallies.

>>> from aci_tallies import mctal
>>> f = mctal.open("mctal")
>>> f.ids, f.histories
>>> t = f[4]                      # Tally: all of it
>>> some = f.read(4, f=slice(0, 10))   # only its first ten cells
>>> t.result.shape                # (f, d, u, s, m, c, e, t), always eight axes
>>> t.integrate("e").result       # summed over energy (MCNP's Total bin)

The axes are MCNP's eight bin dimensions, in the order the file lists
values: cells or surfaces (`f`), direct or flagged (`d`), user (`u`),
segments (`s`), multipliers (`m`), cosines (`c`), energies (`e`) and
times (`t`); `t.axes["e"]` has an axis's bins. MCTAL gives upper bounds
only (`Axis.upper`): the first bin starts at the quantity's minimum. An
axis's Total bin, when MCNP printed one, is its last index.

`t.numbers` has the cell or surface of each `f` bin where MCTAL gives one:
not for groups, the T bin, or bins with several cells at their first
level, and never the levels below (`(8 < 7[2 0 0] < 3)` is just 8). Open
the file with `input=` (the MCNP input) and `t.bins` has what each bin is,
every level included, expanded as MCNP expands the F card:
`str(t.bins[3])` is `"(8<7[1 0 0]<3)"`, `t.bins[3].levels` its structure.

`to_polars()` gives one row per value, with a column per axis of more than
one bin: its index (`f`, `d`, ...), its bounds (`e_low`, `e_high`) where
known, `*_is_total` where it has a Total bin, `number` for the cells or
surfaces and `bin` for what the input says each is; then `result` and
`rel_error`.
"""

from ._native import Bin, Kcode, MctalFile, Tfc
from ._native import MctalInfo as TallyInfo
from ._native import MctalTally as Tally
from ._native import open_mctal as open

DIMS = ("f", "d", "u", "s", "m", "c", "e", "t")

__all__ = ["DIMS", "Bin", "Kcode", "MctalFile", "Tally", "TallyInfo", "Tfc", "open"]
