//! Numbers as MCNP prints them.
//!
//! MCNP writes Fortran `E`/`ES` fields such as `1.23456E-04`. When an
//! exponent needs three digits Fortran drops the `E`: `1.23456-104` is
//! 1.23456e-104, which `str::parse` rejects. [`parse`] reads both, and is
//! the hot loop of every reader, so the usual short form takes a fast path:
//! the digits as an integer `m`, the value `m * 10^p`. When `m < 2^53` and
//! `|p| <= 22` both are exact doubles and one multiplication or division
//! rounds correctly (Clinger's fast path); anything else goes to
//! `str::parse`. Either way the result is the correctly rounded double, the
//! same bits `str::parse` gives for the same number.

/// Exact powers of ten as doubles.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// The number in `tok` (no surrounding whitespace): anything `str::parse`
/// takes, plus Fortran's `E`-less exponents (`1.5-104`, `1.5+104`) and `D`
/// exponents.
pub fn parse(tok: &[u8]) -> Option<f64> {
    match fast(tok) {
        Fast::Value(v) => Some(v),
        Fast::Slow { exponent_at } => slow(tok, exponent_at),
        Fast::Invalid => std::str::from_utf8(tok).ok()?.parse().ok(),
    }
}

enum Fast {
    Value(f64),
    /// A number, but not one for the fast path; its exponent (sign or
    /// letter) starts at `exponent_at`.
    Slow {
        exponent_at: Option<usize>,
    },
    /// Not a number of the form `[+-]digits[.digits][exponent]`, though
    /// `str::parse` may still take it (`inf`, `NaN`).
    Invalid,
}

fn fast(tok: &[u8]) -> Fast {
    let mut i = 0;
    let negative = match tok.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let mut mantissa: u64 = 0;
    let mut digits = 0usize; // significant digits in `mantissa`
    let mut any_digit = false;
    let mut fraction = 0i32; // digits after the point
    let mut seen_point = false;
    while i < tok.len() {
        let b = tok[i];
        if b.is_ascii_digit() {
            any_digit = true;
            if mantissa != 0 || b != b'0' {
                digits += 1;
            }
            if digits > 19 {
                return Fast::Slow { exponent_at: None };
            }
            mantissa = mantissa * 10 + u64::from(b - b'0');
            fraction += i32::from(seen_point);
        } else if b == b'.' && !seen_point {
            seen_point = true;
        } else {
            break;
        }
        i += 1;
    }
    if !any_digit {
        return Fast::Invalid;
    }
    let mut power = -fraction;
    let mut exponent_at = None;
    if i < tok.len() {
        exponent_at = Some(i);
        let mut j = i;
        if matches!(tok[j], b'e' | b'E' | b'd' | b'D') {
            j += 1;
        } else if !matches!(tok[j], b'+' | b'-') {
            return Fast::Invalid;
        }
        let exp_negative = match tok.get(j) {
            Some(b'-') => {
                j += 1;
                true
            }
            Some(b'+') => {
                j += 1;
                false
            }
            _ => false,
        };
        if j == tok.len() {
            return Fast::Invalid;
        }
        let mut exp: i32 = 0;
        for &b in &tok[j..] {
            if !b.is_ascii_digit() {
                return Fast::Invalid;
            }
            exp = exp.saturating_mul(10).saturating_add(i32::from(b - b'0'));
        }
        power = power.saturating_add(if exp_negative { -exp } else { exp });
    }
    let value = if mantissa == 0 {
        0.0
    } else if mantissa < (1 << 53) && (-22..=22).contains(&power) {
        let m = mantissa as f64;
        if power >= 0 {
            m * POW10[power as usize]
        } else {
            m / POW10[(-power) as usize]
        }
    } else {
        return Fast::Slow { exponent_at };
    };
    Fast::Value(if negative { -value } else { value })
}

/// `str::parse`, after giving an `E`-less or `D` exponent an `e`.
fn slow(tok: &[u8], exponent_at: Option<usize>) -> Option<f64> {
    let text = std::str::from_utf8(tok).ok()?;
    let Some(at) = exponent_at else {
        return text.parse().ok();
    };
    let mut buf = String::with_capacity(tok.len() + 1);
    buf.push_str(&text[..at]);
    buf.push('e');
    let rest = &text[at..];
    buf.push_str(rest.strip_prefix(['e', 'E', 'd', 'D']).unwrap_or(rest));
    buf.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn same(tok: &str, want: f64) {
        let got = parse(tok.as_bytes()).unwrap_or_else(|| panic!("{tok:?} did not parse"));
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{tok:?}: {got:e} vs {want:e}"
        );
    }

    #[test]
    fn fortran_forms() {
        same("1.23456E-04", 1.23456e-4);
        same("-1.23456E+00", -1.23456);
        same("1.23456-104", 1.23456e-104);
        same("9.99999+100", 9.99999e100);
        same("1.5D+03", 1.5e3);
        same("0.1234", 0.1234);
        same("-0.00000E+00", -0.0);
        same("0.00000E+00", 0.0);
        same("12.", 12.0);
        same(".5", 0.5);
        same("+3", 3.0);
        same("1e400", f64::INFINITY);
        same("1.0E-400", 0.0);
        same("inf", f64::INFINITY);
        assert!(parse(b"nan").unwrap().is_nan());
        for bad in [
            "", "-", ".", "E5", "1.0E", "1.0-", "1..0", "1.0E+5x", "abc", "1-2-3",
        ] {
            assert_eq!(parse(bad.as_bytes()), None, "{bad:?}");
        }
    }

    #[test]
    fn many_digits() {
        same("12345678901234567890123", 12345678901234567890123.0);
        same(
            "0.000000000000000000001234567",
            0.000000000000000000001234567,
        );
    }

    proptest! {
        /// Every number in the forms MCNP prints parses to `str::parse`'s bits.
        #[test]
        fn like_std(v in proptest::num::f64::NORMAL | proptest::num::f64::SUBNORMAL | proptest::num::f64::ZERO,
                    digits in 0usize..17) {
            for tok in [format!("{v:.digits$E}"), format!("{v:.digits$e}"), format!("{v}"), format!("{v:e}")] {
                let want: f64 = tok.parse().unwrap();
                prop_assert_eq!(parse(tok.as_bytes()).map(f64::to_bits), Some(want.to_bits()), "{}", tok);
            }
        }

        /// The `E`-less form reads as the same number with an `E`.
        #[test]
        fn fortran_three_digit_exponent(m in 100000u32..1000000, e in 100i32..330, neg in any::<bool>()) {
            let mantissa = format!("{}.{:05}", m / 100000, m % 100000);
            let sign = if neg { '-' } else { '+' };
            let want: f64 = format!("{mantissa}e{sign}{e}").parse().unwrap();
            let got = parse(format!("{mantissa}{sign}{e}").as_bytes());
            prop_assert_eq!(got.map(f64::to_bits), Some(want.to_bits()));
        }
    }
}
