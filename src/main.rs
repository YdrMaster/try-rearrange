use cuda::{Ptx, Symbol};
use ndarray_layout::Endian::BigEndian;
use std::ffi::{c_uint, CString};

type Arr = ndarray_layout::ArrayLayout<4>;

const CODE: &str = include_str!("rearrange.cu");

fn main() {
    let Some(Symbol::Global(name)) = Symbol::search(CODE).next() else {
        return;
    };
    let name = CString::new(name).unwrap();

    cuda::init().unwrap();
    let device = cuda::Device::new(0);
    let cc = device.compute_capability();
    let (ptx, log) = Ptx::compile(CODE, cc);

    if !log.is_empty() {
        println!("{log}")
    }
    let ptx = ptx.unwrap();
    device.context().apply(|ctx| {
        let module = ctx.load(&ptx);
        let kernel = module.get_kernel(&name);

        const D: usize = 1 << 14;
        const E: usize = 1 << 3;
        let mut src = ctx.malloc::<u8>(D * D * E);
        let mut dst = ctx.malloc::<u8>(D * D * E);

        let dst = dst.as_mut_ptr();
        let src = src.as_ptr();

        let mem = Arr::new_contiguous(&[D, D], BigEndian, E);

        for i in 0..=3 {
            let step = 1 << i;
            let src_ = mem.clone().slice(1, 0, step, D / 8);
            let dst_ = mem.clone().slice(1, 0, 1, D / 8);

            let &[dst_sy, dst_sx] = dst_.strides() else {
                unreachable!()
            };
            let &[src_sy, src_sx] = src_.strides() else {
                unreachable!()
            };

            let params = cuda::params![dst, dst_sy, dst_sx, src, src_sy, src_sx];
            kernel.launch(
                (D as c_uint, ((D / 8) >> 10) as c_uint),
                1024,
                params.as_ptr(),
                0,
                None,
            )
        }
    })
}
