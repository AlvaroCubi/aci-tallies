//! The MCNP input a tally file came from.
//!
//! A tally file leaves out what its tallies mean in detail (meshtal headers
//! round mesh bounds and omit `TR=`; MCTAL files give only the number of
//! some bins), and the input has it. It is parsed once, with migjorn, and
//! each reader takes what it needs from it.

use std::io;
use std::path::Path;

pub struct Input {
    model: migjorn::Model,
}

impl Input {
    /// The input `src`.
    pub fn parse(src: &str) -> Input {
        Input {
            model: migjorn::Model::parse(src),
        }
    }

    /// The input at `path`, read as UTF-8 with invalid bytes (say, Latin-1
    /// comments) replaced.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Input> {
        let bytes = std::fs::read(path)?;
        Ok(Input::parse(&String::from_utf8_lossy(&bytes)))
    }

    pub fn model(&self) -> &migjorn::Model {
        &self.model
    }
}
