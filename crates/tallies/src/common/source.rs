//! Where a file's bytes come from: a file on disk, or bytes in memory.
//!
//! A reader keeps no file handle open. Every read opens the file again and
//! checks that its size and modification time are what they were when it
//! was indexed, so a file rewritten since (by a rerun of MCNP, say) is
//! reported as [`ReadError::Changed`] instead of read with a stale index.
//! Within one read the opened file is shared by every thread, each reading
//! its own part with positioned reads.

use std::borrow::Cow;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// Why a read failed.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The file's size or modification time is not what it was when
    /// indexed: open it again.
    #[error("{} changed since it was indexed", .0.display())]
    Changed(PathBuf),
    #[error(
        "no tally {} (tallies: {:?})",
        .id.map_or("at all".to_string(), |n| n.to_string()),
        .tallies
    )]
    NoTally { id: Option<u32>, tallies: Vec<u32> },
    /// The tally's section could not be read (see the reader's `failed`).
    #[error("tally {id}: {reason}")]
    Unsupported { id: u32, reason: String },
    #[error("tally {id}: {axis} index {index} of {count}")]
    NoBin {
        id: u32,
        axis: &'static str,
        index: usize,
        count: usize,
    },
}

impl From<ReadError> for io::Error {
    fn from(e: ReadError) -> io::Error {
        match e {
            ReadError::Io(e) => e,
            e => io::Error::other(e),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    len: u64,
    modified: Option<SystemTime>,
}

impl Stamp {
    fn of(file: &File) -> io::Result<Stamp> {
        let meta = file.metadata()?;
        Ok(Stamp {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// A file's bytes, read by parts.
#[derive(Clone, Debug)]
pub struct Source(Origin);

#[derive(Clone, Debug)]
enum Origin {
    File { path: PathBuf, stamp: Stamp },
    Bytes(Arc<[u8]>),
}

impl Source {
    /// The file at `path`, as it is now.
    pub fn file(path: &Path) -> io::Result<Source> {
        let file = File::open(path)?;
        Ok(Source(Origin::File {
            path: path.to_path_buf(),
            stamp: Stamp::of(&file)?,
        }))
    }

    pub fn bytes(bytes: impl Into<Arc<[u8]>>) -> Source {
        Source(Origin::Bytes(bytes.into()))
    }

    /// The file's path (`None` for bytes in memory).
    pub fn path(&self) -> Option<&Path> {
        match &self.0 {
            Origin::File { path, .. } => Some(path),
            Origin::Bytes(_) => None,
        }
    }

    pub fn len(&self) -> u64 {
        match &self.0 {
            Origin::File { stamp, .. } => stamp.len,
            Origin::Bytes(b) => b.len() as u64,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the file still has the size and modification time it had
    /// when this source was made (always true for bytes).
    pub fn is_current(&self) -> bool {
        match &self.0 {
            Origin::File { path, stamp } => File::open(path)
                .and_then(|f| Stamp::of(&f))
                .is_ok_and(|s| s == *stamp),
            Origin::Bytes(_) => true,
        }
    }

    /// Open for one read, checking that the file has not changed.
    pub(crate) fn open(&self) -> Result<Opened<'_>, ReadError> {
        match &self.0 {
            Origin::File { path, stamp } => {
                let file = File::open(path)?;
                if Stamp::of(&file)? != *stamp {
                    return Err(ReadError::Changed(path.clone()));
                }
                Ok(Opened::File {
                    file,
                    path,
                    len: stamp.len,
                })
            }
            Origin::Bytes(b) => Ok(Opened::Bytes(b)),
        }
    }
}

/// A source opened for one read; `Sync`, so threads share it.
pub(crate) enum Opened<'a> {
    File {
        file: File,
        path: &'a Path,
        len: u64,
    },
    Bytes(&'a [u8]),
}

impl Opened<'_> {
    pub fn len(&self) -> u64 {
        match self {
            Opened::File { len, .. } => *len,
            Opened::Bytes(b) => b.len() as u64,
        }
    }

    /// Bytes `range`, which must lie within the file as indexed: a file
    /// that turns out shorter has changed.
    pub fn read(&self, range: Range<u64>) -> Result<Cow<'_, [u8]>, ReadError> {
        let want = range.end.saturating_sub(range.start);
        let bytes = self.read_upto(range)?;
        if (bytes.len() as u64) < want {
            let path = match self {
                Opened::File { path, .. } => path.to_path_buf(),
                Opened::Bytes(_) => PathBuf::new(),
            };
            return Err(ReadError::Changed(path));
        }
        Ok(bytes)
    }

    /// Bytes `range`, or fewer at the end of the file.
    pub fn read_upto(&self, range: Range<u64>) -> io::Result<Cow<'_, [u8]>> {
        match self {
            Opened::Bytes(b) => {
                let end = (range.end as usize).min(b.len());
                let start = (range.start as usize).min(end);
                Ok(Cow::Borrowed(&b[start..end]))
            }
            Opened::File { file, .. } => read_at(file, range).map(Cow::Owned),
        }
    }
}

/// Bytes `range` of `file` (fewer at its end), by positioned reads.
fn read_at(file: &File, range: Range<u64>) -> io::Result<Vec<u8>> {
    let mut buf = vec![0; range.end.saturating_sub(range.start) as usize];
    let mut n = 0;
    while n < buf.len() {
        let at = range.start + n as u64;
        #[cfg(unix)]
        let r = std::os::unix::fs::FileExt::read_at(file, &mut buf[n..], at);
        #[cfg(windows)]
        let r = std::os::windows::fs::FileExt::seek_read(file, &mut buf[n..], at);
        match r {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    buf.truncate(n);
    Ok(buf)
}
