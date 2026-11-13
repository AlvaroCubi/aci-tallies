//! Values below 1e-99, which Fortran prints without the `E`
//! (`2.50000-113`): `tests/fixtures/tiny_values.mcnp` scales a pencil
//! beam's flux (20 cm chord / 8000 cm³ voxels) by 1e-110.

mod common;

use aci_tallies::meshtal::{OutFormat, Reader};

#[test]
fn three_digit_exponents_read_exactly() {
    let r = Reader::open(common::fixture("tiny_values.msht")).unwrap();
    for (id, format) in [(1004, OutFormat::Col), (1014, OutFormat::Ij)] {
        let t = r.read(id).unwrap();
        assert_eq!(t.out_format, format);
        assert!(t.warnings.is_empty(), "tally {id}: {:?}", t.warnings);
        for &v in &t.result {
            assert_eq!(v, 2.5e-113, "tally {id}");
        }
    }
}
