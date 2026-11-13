//! Problems found while reading that don't stop it.
//!
//! Tally files can stand for days of computing, so a problem in one part of
//! one (an unexpected line, an unreadable number) never stops the rest from
//! being read: it becomes a [`Warning`] on what was read, and the values it
//! affects are `NaN`.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    /// The line of the file it is about (from 1), when there is one.
    pub line: Option<u64>,
    pub message: String,
}

impl Warning {
    pub fn new(message: impl Into<String>) -> Warning {
        Warning {
            line: None,
            message: message.into(),
        }
    }

    /// About line `line` (from 1).
    pub fn at(line: u64, message: impl Into<String>) -> Warning {
        Warning {
            line: Some(line),
            message: message.into(),
        }
    }
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "line {line}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}
