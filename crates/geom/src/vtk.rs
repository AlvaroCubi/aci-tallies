//! VTK XML export of a [`Geometry`] with cell data.
//!
//! Rectangular meshes are written as `RectilinearGrid` (`.vtr`), cylindrical
//! and transformed (rotated, `TR=`) rectangular ones as `StructuredGrid`
//! (`.vts`). Data goes in the appended section as
//! little-endian binary, zlib-compressed by default (`vtkZLibDataCompressor`,
//! as PyVista writes): mesh results are mostly zeros and shrink by one to
//! two orders of magnitude. ParaView and PyVista read both forms.
//!
//! Cell values are given in this crate's flat voxel order (inner axis
//! fastest) and reordered here to VTK's (first axis fastest).
//!
//! A cylindrical mesh often has a single theta bin (a full revolution),
//! which would draw as a flat sheet. [`VtkOptions::max_theta_step`]
//! subdivides theta bins for display; every sub-cell repeats its voxel's
//! values, so the data is unchanged.
//!
//! [`layout`] exposes the grid itself (coordinates or points, and which
//! voxel each cell shows) for callers that build VTK objects in memory,
//! such as PyVista, so they match the files written here exactly.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::path::Path;

use crate::Geometry;

/// Values of one named cell-data array, one per voxel in flat index order.
#[derive(Debug, Clone, Copy)]
pub enum CellValues<'a> {
    F64(&'a [f64]),
    I32(&'a [i32]),
}

impl CellValues<'_> {
    fn len(&self) -> usize {
        match self {
            CellValues::F64(v) => v.len(),
            CellValues::I32(v) => v.len(),
        }
    }
}

/// One named cell-data array.
#[derive(Debug, Clone, Copy)]
pub struct CellArray<'a> {
    pub name: &'a str,
    pub values: CellValues<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct VtkOptions {
    /// Largest theta step, in revolutions, drawn for a cylindrical mesh;
    /// coarser theta bins are split into equal sub-cells. Default 1/72 (5°).
    pub max_theta_step: f64,
    /// Compress data with zlib. Default true.
    pub compress: bool,
}

impl Default for VtkOptions {
    fn default() -> Self {
        VtkOptions {
            max_theta_step: 1.0 / 72.0,
            compress: true,
        }
    }
}

/// File extension VTK expects for this geometry: `"vtr"` or `"vts"`.
pub fn extension(geometry: &Geometry) -> &'static str {
    match geometry {
        Geometry::Rectangular {
            transform: None, ..
        } => "vtr",
        Geometry::Rectangular { .. } | Geometry::Cylindrical { .. } => "vts",
    }
}

/// Write `geometry` and `arrays` to `path` (use [`extension`] for its suffix).
pub fn write_file(
    path: impl AsRef<Path>,
    geometry: &Geometry,
    arrays: &[CellArray],
    options: &VtkOptions,
) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out = io::BufWriter::new(file);
    write(&mut out, geometry, arrays, options)?;
    out.flush()
}

/// A geometry laid out as a VTK structured dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum Grid {
    /// A rectangular mesh, as `vtkRectilinearGrid` coordinates.
    Rectilinear {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
    },
    /// A cylindrical or transformed rectangular mesh, as a
    /// `vtkStructuredGrid`: `dims` points along the grid axes and every
    /// point, first axis fastest. A rectangular mesh's grid axes are its
    /// local (x, y, z); a cylinder's are (r, theta, z), in that order because
    /// it is right-handed (e_r x e_theta = e_z), so cells have positive
    /// volume; (r, z, theta) would turn them inside out.
    Structured {
        dims: [usize; 3],
        points: Vec<[f64; 3]>,
    },
}

impl Grid {
    /// Number of points along each grid axis.
    pub fn dims(&self) -> [usize; 3] {
        match self {
            Grid::Rectilinear { x, y, z } => [x.len(), y.len(), z.len()],
            Grid::Structured { dims, .. } => *dims,
        }
    }
}

/// A [`Grid`] plus where each of its cells' values come from.
#[derive(Debug, Clone, PartialEq)]
pub struct GridLayout {
    pub grid: Grid,
    /// Flat voxel index of every VTK cell, in VTK cell order (first grid
    /// axis fastest): a cell's value is `values[cell_voxels[cell]]`. A voxel
    /// appears several times when its theta bin was subdivided.
    pub cell_voxels: Vec<usize>,
}

/// Lay out `geometry` as a VTK grid, splitting a cylinder's theta bins into
/// steps no coarser than `max_theta_step` revolutions (`0` keeps them).
pub fn layout(geometry: &Geometry, max_theta_step: f64) -> GridLayout {
    let (ni, nj, nk) = geometry.dims();
    match geometry {
        Geometry::Rectangular {
            x_bounds,
            y_bounds,
            z_bounds,
            transform,
        } => {
            // A reflection would turn every cell inside out (negative
            // volume in VTK), so its grid runs along x backwards.
            let flip = transform.is_some_and(|t| t.is_reflection());
            GridLayout {
                // VTK cell (a, b, c), a fastest = voxel (i, j, k) = (a, b, c),
                // or (ni - 1 - a, b, c) when flipped.
                cell_voxels: vtk_order(ni, nj, nk, |a, b, c| {
                    geometry.index(if flip { ni - 1 - a } else { a }, b, c)
                }),
                grid: match transform {
                    None => Grid::Rectilinear {
                        x: x_bounds.clone(),
                        y: y_bounds.clone(),
                        z: z_bounds.clone(),
                    },
                    Some(t) => {
                        let mut points = Vec::with_capacity((ni + 1) * (nj + 1) * (nk + 1));
                        for &z in z_bounds {
                            for &y in y_bounds {
                                let push = |&x: &f64| t.to_world([x, y, z]);
                                if flip {
                                    points.extend(x_bounds.iter().rev().map(push));
                                } else {
                                    points.extend(x_bounds.iter().map(push));
                                }
                            }
                        }
                        Grid::Structured {
                            dims: [ni + 1, nj + 1, nk + 1],
                            points,
                        }
                    }
                },
            }
        }
        Geometry::Cylindrical {
            r_bounds,
            z_bounds,
            theta_bounds,
            ..
        } => {
            let frame = geometry
                .cyl_frame()
                .expect("cylindrical geometry has a frame");
            let (thetas, parent) = subdivide(theta_bounds, max_theta_step);
            let nt = thetas.len() - 1;
            let mut points = Vec::with_capacity((ni + 1) * (nt + 1) * (nk + 1));
            for &z in z_bounds {
                for &t in &thetas {
                    for &r in r_bounds {
                        points.push(frame.to_world(r, z, t));
                    }
                }
            }
            GridLayout {
                // Grid axes (r, theta, z); theta sub-cell s belongs to
                // voxel theta bin parent[s].
                cell_voxels: vtk_order(ni, nt, nj, |a, b, c| geometry.index(a, c, parent[b])),
                grid: Grid::Structured {
                    dims: [ni + 1, nt + 1, nj + 1],
                    points,
                },
            }
        }
    }
}

/// Write `geometry` and `arrays` as a VTK XML document to `out`.
pub fn write(
    out: &mut impl Write,
    geometry: &Geometry,
    arrays: &[CellArray],
    options: &VtkOptions,
) -> io::Result<()> {
    let n_voxels = geometry.n_voxels();
    for array in arrays {
        if array.values.len() != n_voxels {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cell array {:?} has {} values, the mesh has {} voxels",
                    array.name,
                    array.values.len(),
                    n_voxels
                ),
            ));
        }
    }

    let GridLayout { grid, cell_voxels } = layout(geometry, options.max_theta_step);
    let [px, py, pz] = grid.dims();
    let extent = format!("0 {} 0 {} 0 {}", px - 1, py - 1, pz - 1);
    let mut doc = Document {
        compress: options.compress,
        ..Document::default()
    };
    match grid {
        Grid::Rectilinear { x, y, z } => {
            doc.open("RectilinearGrid", &extent);
            doc.cell_data(arrays, &cell_voxels);
            doc.xml.push_str("<Coordinates>\n");
            for (name, bounds) in [("x", x), ("y", y), ("z", z)] {
                doc.array_f64(name, 1, bounds.into_iter());
            }
            doc.xml.push_str("</Coordinates>\n");
            doc.close("RectilinearGrid");
        }
        Grid::Structured { points, .. } => {
            doc.open("StructuredGrid", &extent);
            doc.cell_data(arrays, &cell_voxels);
            doc.xml.push_str("<Points>\n");
            doc.array_f64("Points", 3, points.into_iter().flatten());
            doc.xml.push_str("</Points>\n");
            doc.close("StructuredGrid");
        }
    }
    doc.finish(out)
}

/// Flat voxel index of every VTK cell, in VTK order (first axis fastest).
fn vtk_order(
    na: usize,
    nb: usize,
    nc: usize,
    voxel: impl Fn(usize, usize, usize) -> usize,
) -> Vec<usize> {
    let mut order = Vec::with_capacity(na * nb * nc);
    for c in 0..nc {
        for b in 0..nb {
            for a in 0..na {
                order.push(voxel(a, b, c));
            }
        }
    }
    order
}

/// Split each bin of `bounds` into equal steps no larger than `max_step`.
/// Returns the new bounds and, for each new bin, its original bin.
fn subdivide(bounds: &[f64], max_step: f64) -> (Vec<f64>, Vec<usize>) {
    let mut out = vec![bounds[0]];
    let mut parent = Vec::new();
    for (bin, w) in bounds.windows(2).enumerate() {
        let width = w[1] - w[0];
        let n = if max_step > 0.0 {
            ((width / max_step).ceil() as usize).max(1)
        } else {
            1
        };
        for s in 1..=n {
            out.push(if s == n {
                w[1]
            } else {
                w[0] + width * s as f64 / n as f64
            });
            parent.push(bin);
        }
    }
    (out, parent)
}

/// Size of the uncompressed blocks of a compressed data array.
const BLOCK: usize = 1 << 15;

/// A VTK XML document with an appended binary data section.
#[derive(Default)]
struct Document {
    xml: String,
    appended: Vec<u8>,
    compress: bool,
}

impl Document {
    fn open(&mut self, kind: &str, extent: &str) {
        let compressor = if self.compress {
            " compressor=\"vtkZLibDataCompressor\""
        } else {
            ""
        };
        let _ = write!(
            self.xml,
            "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"{kind}\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{compressor}>\n\
             <{kind} WholeExtent=\"{extent}\">\n\
             <Piece Extent=\"{extent}\">\n"
        );
    }

    fn close(&mut self, kind: &str) {
        let _ = write!(self.xml, "</Piece>\n</{kind}>\n");
    }

    fn cell_data(&mut self, arrays: &[CellArray], order: &[usize]) {
        self.xml.push_str("<CellData>\n");
        for array in arrays {
            match array.values {
                CellValues::F64(v) => self.array_f64(array.name, 1, order.iter().map(|&n| v[n])),
                CellValues::I32(v) => {
                    self.begin_array("Int32", array.name, 1);
                    let raw: Vec<u8> = order.iter().flat_map(|&n| v[n].to_le_bytes()).collect();
                    self.push_data(&raw);
                }
            }
        }
        self.xml.push_str("</CellData>\n");
    }

    fn array_f64(&mut self, name: &str, components: usize, values: impl Iterator<Item = f64>) {
        self.begin_array("Float64", name, components);
        let raw: Vec<u8> = values.flat_map(f64::to_le_bytes).collect();
        self.push_data(&raw);
    }

    fn begin_array(&mut self, ty: &str, name: &str, components: usize) {
        let _ = writeln!(
            self.xml,
            "<DataArray type=\"{ty}\" Name=\"{}\" NumberOfComponents=\"{components}\" format=\"appended\" offset=\"{}\"/>",
            escape(name),
            self.appended.len()
        );
    }

    /// Append one array's bytes. Uncompressed: `[len: u64] data`.
    /// Compressed: `[n_blocks, block_size, last_block_size, compressed
    /// size of each block: u64...]` then the zlib streams of each block.
    fn push_data(&mut self, raw: &[u8]) {
        let u64le = |x: usize| (x as u64).to_le_bytes();
        if !self.compress {
            self.appended.extend_from_slice(&u64le(raw.len()));
            self.appended.extend_from_slice(raw);
            return;
        }
        let blocks: Vec<Vec<u8>> = raw.chunks(BLOCK).map(zlib).collect();
        let last = match raw.len() % BLOCK {
            0 if !raw.is_empty() => BLOCK,
            r => r,
        };
        self.appended.extend_from_slice(&u64le(blocks.len()));
        self.appended.extend_from_slice(&u64le(BLOCK));
        self.appended.extend_from_slice(&u64le(last));
        for b in &blocks {
            self.appended.extend_from_slice(&u64le(b.len()));
        }
        for b in &blocks {
            self.appended.extend_from_slice(b);
        }
    }

    fn finish(self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(self.xml.as_bytes())?;
        out.write_all(b"<AppendedData encoding=\"raw\">\n_")?;
        out.write_all(&self.appended)?;
        out.write_all(b"\n</AppendedData>\n</VTKFile>\n")
    }
}

fn zlib(data: &[u8]) -> Vec<u8> {
    use flate2::{write::ZlibEncoder, Compression};
    let mut e = ZlibEncoder::new(Vec::with_capacity(data.len() / 4), Compression::fast());
    e.write_all(data).expect("writing to memory");
    e.finish().expect("writing to memory")
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdivide_keeps_fine_bins_and_splits_coarse_ones() {
        let (b, p) = subdivide(&[0.0, 0.01, 1.0], 0.25);
        assert_eq!(b, vec![0.0, 0.01, 0.2575, 0.505, 0.7525, 1.0]);
        assert_eq!(p, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn rectangular_cell_data_is_reordered_first_axis_fastest() {
        let g = Geometry::Rectangular {
            x_bounds: vec![0.0, 1.0, 2.0],
            y_bounds: vec![0.0, 1.0],
            z_bounds: vec![0.0, 1.0, 2.0],
            transform: None,
        };
        // Flat order (x, y, z), z fastest: value = 10*i + k.
        let values: Vec<i32> = (0..4)
            .map(|n| {
                let (i, _, k) = g.ijk(n);
                10 * i as i32 + k as i32
            })
            .collect();
        let mut buf = Vec::new();
        write(
            &mut buf,
            &g,
            &[CellArray {
                name: "v",
                values: CellValues::I32(&values),
            }],
            &VtkOptions {
                compress: false,
                ..VtkOptions::default()
            },
        )
        .unwrap();
        let start = buf.windows(2).position(|w| w == b"\n_").unwrap() + 2;
        let block = &buf[start + 8..start + 8 + 16];
        let got: Vec<i32> = block
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // VTK order: x fastest -> (i,k) = (0,0), (1,0), (0,1), (1,1).
        assert_eq!(got, vec![0, 10, 1, 11]);
        assert!(String::from_utf8_lossy(&buf).contains("WholeExtent=\"0 2 0 1 0 2\""));
    }

    #[test]
    fn cylindrical_layout_subdivides_theta_and_covers_every_voxel() {
        let g = Geometry::Cylindrical {
            r_bounds: vec![0.0, 1.0, 2.0],
            z_bounds: vec![0.0, 5.0],
            theta_bounds: vec![0.0, 0.25, 1.0],
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            vec: [1.0, 0.0, 0.0],
        };
        let GridLayout { grid, cell_voxels } = layout(&g, 0.25);
        // Theta bins of 0.25 and 0.75 rev become 1 + 3 sub-bins.
        assert_eq!(grid.dims(), [3, 5, 2]);
        assert_eq!(cell_voxels.len(), 2 * 4);
        let mut per_voxel = vec![0; g.n_voxels()];
        for &v in &cell_voxels {
            per_voxel[v] += 1;
        }
        // r fastest, then theta: (r0,t0) (r1,t0) (r0,t1') ... ; t1 has 3 sub-cells.
        assert_eq!(per_voxel, vec![1, 3, 1, 3]);
        assert_eq!(&cell_voxels[..4], &[0, 2, 1, 3]);
        let Grid::Structured { points, .. } = grid else {
            panic!("cylinder should be structured")
        };
        // Point (r=2, theta=0.25, z=5): index r + 3*(theta + 5*z).
        let p = points[2 + 3 * (1 + 5)];
        assert!((p[0]).abs() < 1e-12 && (p[1] - 2.0).abs() < 1e-12 && p[2] == 5.0);
        // Without subdivision the grid follows the voxel bins.
        assert_eq!(layout(&g, 0.0).grid.dims(), [3, 3, 2]);
    }

    #[test]
    fn transformed_rectangular_mesh_is_a_structured_grid() {
        // Quarter turn about Z: local x along world +Y, shifted by (10, 0, 0).
        let t = crate::Transform {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [10.0, 0.0, 0.0],
        };
        let g = Geometry::Rectangular {
            x_bounds: vec![0.0, 1.0, 3.0],
            y_bounds: vec![0.0, 2.0],
            z_bounds: vec![0.0, 1.0],
            transform: Some(t),
        };
        assert_eq!(extension(&g), "vts");
        let GridLayout { grid, cell_voxels } = layout(&g, 0.0);
        assert_eq!(cell_voxels, vec![0, 1]);
        let Grid::Structured { dims, points } = grid else {
            panic!("a rotated mesh can't be rectilinear")
        };
        assert_eq!(dims, [3, 2, 2]);
        // Point (x=3, y=0, z=0) is the third, x fastest.
        assert_eq!(points[2], [10.0, 3.0, 0.0]);
        assert_eq!(points[3], [8.0, 0.0, 0.0]);
    }

    #[test]
    fn reflected_rectangular_mesh_runs_backwards_along_x() {
        // x' = +X, y' = +Y, z' = -Z: a mirror through z = 0.
        let t = crate::Transform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            translation: [0.0; 3],
        };
        let g = Geometry::Rectangular {
            x_bounds: vec![0.0, 1.0, 3.0],
            y_bounds: vec![0.0, 2.0],
            z_bounds: vec![0.0, 1.0],
            transform: Some(t),
        };
        let GridLayout { grid, cell_voxels } = layout(&g, 0.0);
        // VTK cell 0 is voxel 1 (x from 3 down to 1).
        assert_eq!(cell_voxels, vec![1, 0]);
        let Grid::Structured { points, .. } = grid else {
            panic!("structured grid expected")
        };
        assert_eq!(
            points[..3],
            [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
        // Cell 0 spans +x (here -x), +y, -z: a right-handed hexahedron.
        let edge = |n: usize| crate::sub(points[n], points[0]);
        let (a, b, c) = (edge(1), edge(3), edge(6));
        assert!(crate::dot(crate::cross(a, b), c) > 0.0);
    }

    #[test]
    fn wrong_length_is_rejected() {
        let g = Geometry::Rectangular {
            x_bounds: vec![0.0, 1.0],
            y_bounds: vec![0.0, 1.0],
            z_bounds: vec![0.0, 1.0],
            transform: None,
        };
        let err = write(
            &mut Vec::new(),
            &g,
            &[CellArray {
                name: "v",
                values: CellValues::F64(&[1.0, 2.0]),
            }],
            &VtkOptions::default(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn compressed_blocks_round_trip() {
        use std::io::Read;
        // 3 full blocks and a partial one, mostly zeros.
        let raw: Vec<u8> = (0..3 * BLOCK + 100).map(|i| (i % 997 == 0) as u8).collect();
        let mut doc = Document {
            compress: true,
            ..Document::default()
        };
        doc.push_data(&raw);
        let a = &doc.appended;
        let word = |i: usize| u64::from_le_bytes(a[8 * i..8 * i + 8].try_into().unwrap()) as usize;
        assert_eq!((word(0), word(1), word(2)), (4, BLOCK, 100));
        let mut offset = 8 * (3 + 4);
        let mut back = Vec::new();
        for b in 0..4 {
            let len = word(3 + b);
            flate2::read::ZlibDecoder::new(&a[offset..offset + len])
                .read_to_end(&mut back)
                .unwrap();
            offset += len;
        }
        assert_eq!(offset, a.len());
        assert_eq!(back, raw);
        assert!(a.len() < raw.len() / 10);
    }
}
