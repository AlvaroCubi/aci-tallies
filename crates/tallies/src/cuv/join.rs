//! A CuV mesh from its meshinfo and cuvmsh together, and voxel fluxes.

use std::io;
use std::ops::Range;
use std::path::Path;

use ndarray::{Array3, Array5};

use super::model::{CuvInfo, CuvPart, Pairs, Spectra};
use super::{Cuvmsh, Meshinfo};
use crate::common::{Axis, Input, ReadError};
use crate::meshtal::model::{Tally, TallyInfo};
use crate::meshtal::OutFormat;

/// A run's `cuvmsh` and `meshinfo`, read together: each pair's material,
/// density and spectrum.
pub struct CuvMesh {
    pub cuvmsh: Cuvmsh,
    pub meshinfo: Meshinfo,
}

impl CuvMesh {
    /// Index both files, with each mesh's geometry from `input` when
    /// given (see [`Cuvmsh::open_with_input`]).
    pub fn open(
        cuvmsh: impl AsRef<Path>,
        meshinfo: impl AsRef<Path>,
        input: Option<&Input>,
    ) -> io::Result<CuvMesh> {
        let (cuvmsh, meshinfo) = (cuvmsh.as_ref(), meshinfo.as_ref());
        let (cuvmsh, meshinfo) = rayon::join(
            || match input {
                Some(i) => Cuvmsh::open_with_input(cuvmsh, i),
                None => Cuvmsh::open(cuvmsh),
            },
            || match input {
                Some(i) => Meshinfo::open_with_input(meshinfo, i),
                None => Meshinfo::open(meshinfo),
            },
        );
        Ok(CuvMesh {
            cuvmsh: cuvmsh?,
            meshinfo: meshinfo?,
        })
    }

    /// The meshes both files have, as the cuvmsh describes them.
    pub fn tallies(&self) -> impl Iterator<Item = &CuvInfo> {
        self.cuvmsh
            .tallies()
            .filter(|t| self.meshinfo.tally(t.id).is_ok())
    }

    /// Mesh `id`, as the cuvmsh describes it, once both files are found
    /// to describe the same mesh.
    pub fn tally(&self, id: u32) -> Result<&CuvInfo, ReadError> {
        let c = self.cuvmsh.tally(id)?;
        let m = self.meshinfo.tally(id)?;
        if c.geometry != m.geometry {
            return Err(ReadError::Unsupported {
                id,
                reason: "the cuvmsh and the meshinfo have different geometries: are they \
                         from the same run?"
                    .into(),
            });
        }
        Ok(c)
    }

    /// Every pair of mesh `id`.
    pub fn read(&self, id: u32) -> Result<CuvPart, ReadError> {
        let n = self.tally(id)?.n_voxels;
        self.read_voxels(id, 0..n)
    }

    /// The pairs of voxels `voxels` of mesh `id` from both files, which
    /// must list the same pairs.
    pub fn read_voxels(&self, id: u32, voxels: Range<usize>) -> Result<CuvPart, ReadError> {
        self.tally(id)?;
        let (spectra, decomposition) = rayon::join(
            || self.cuvmsh.read_voxels(id, voxels.clone()),
            || self.meshinfo.read_voxels(id, voxels.clone()),
        );
        let (spectra, decomposition) = (spectra?, decomposition?);
        same_pairs(&decomposition.pairs, &spectra.pairs)
            .map_err(|reason| ReadError::Unsupported { id, reason })?;
        Ok(CuvPart {
            decomposition,
            spectra,
        })
    }
}

/// Whether the meshinfo's pairs `m` are the cuvmsh's `c`, at the files'
/// print precision.
fn same_pairs(m: &Pairs, c: &Pairs) -> Result<(), String> {
    let mismatch = |what: String| {
        Err(format!(
            "{what} in the meshinfo and the cuvmsh: are they from the same run?"
        ))
    };
    if m.len() != c.len() {
        return mismatch(format!("{} pairs and {}", m.len(), c.len()));
    }
    for (n, (vm, vc)) in m.voxel_volume.iter().zip(&c.voxel_volume).enumerate() {
        if (vm - vc).abs() > 1e-5 * vm.abs().max(vc.abs()) {
            let v = m.voxels.start + n + 1;
            return mismatch(format!("voxel {v} has volumes {vm} and {vc}"));
        }
    }
    for p in 0..m.len() {
        let differ = m.voxel[p] != c.voxel[p]
            || m.cell[p] != c.cell[p]
            || (m.fraction[p] - c.fraction[p]).abs() > 1.1e-5;
        if differ {
            return mismatch(format!(
                "voxel {} has cell {} ({}) and cell {} ({})",
                m.voxel[p] + 1,
                m.cell[p],
                m.fraction[p],
                c.cell[p],
                c.fraction[p]
            ));
        }
    }
    Ok(())
}

impl Spectra {
    /// The flux of each voxel read, as a meshtal tally of mesh `info`: the
    /// sum over its cells of their flux times their share of its volume,
    /// which is what an FMESH without `out=cuv` scores. Voxels not read
    /// are `NaN`. A mesh with more than one energy bin gets a Total bin,
    /// the same sum of the pairs' total fluxes.
    ///
    /// Errors are combined as if the cells' fluxes were independent. They
    /// are not (the same histories cross several cells of a voxel), so the
    /// voxel's error is only an estimate, usually low.
    pub fn voxel_flux(&self, info: &CuvInfo) -> Tally {
        let geometry = &info.geometry;
        let (ni, nj, nk) = geometry.dims();
        let voxels = ni * nj * nk;
        let n = self.flux.ncols();
        let bounds = info
            .energy
            .as_ref()
            .and_then(|e| e.bounds().map(<[f64]>::to_vec))
            .unwrap_or_else(|| vec![0.0, 1e36]);
        let energy = Axis::mcnp(bounds);
        let ne = energy.len();
        let mut value = vec![f64::NAN; ne * voxels];
        let mut var = vec![f64::NAN; ne * voxels];
        for v in self.pairs.voxels.clone() {
            for e in 0..ne {
                value[e * voxels + v] = 0.0;
                var[e * voxels + v] = 0.0;
            }
        }
        for p in 0..self.pairs.len() {
            let v = self.pairs.voxel[p] as usize;
            let f = self.pairs.fraction[p];
            let mut add = |e: usize, flux: f64, rel: f64| {
                let x = f * flux;
                value[e * voxels + v] += x;
                var[e * voxels + v] += (x * rel).powi(2);
            };
            for e in 0..n {
                add(e, self.flux[[p, e]], self.rel_error[[p, e]]);
            }
            if ne > n {
                add(n, self.total[p], self.total_error[p]);
            }
        }
        let rel: Vec<f64> = value
            .iter()
            .zip(&var)
            .map(|(&x, &s2)| {
                if x > 0.0 {
                    s2.sqrt() / x
                } else if x.is_nan() {
                    f64::NAN
                } else {
                    0.0
                }
            })
            .collect();
        let mut volume = vec![f64::NAN; voxels];
        for (k, v) in self.pairs.voxels.clone().enumerate() {
            volume[v] = self.pairs.voxel_volume[k];
        }
        let shape = (ne, 1, ni, nj, nk);
        let tally_info = TallyInfo {
            id: info.id,
            particle: info.particle.clone().unwrap_or_default(),
            comment: info.comment.clone(),
            geometry: geometry.clone(),
            geometry_from_input: info.geometry_from_input,
            energy_weighted: None,
            energy,
            time: Axis::mcnp(vec![0.0, 1e36]),
            out_format: OutFormat::Cuv,
            warnings: Vec::new(),
        };
        Tally::from_info(
            tally_info,
            Array5::from_shape_vec(shape, value).expect("one value per index"),
            Array5::from_shape_vec(shape, rel).expect("one value per index"),
            Some(Array3::from_shape_vec((ni, nj, nk), volume).expect("one volume per voxel")),
            self.warnings.clone(),
        )
    }
}
