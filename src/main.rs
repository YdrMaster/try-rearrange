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

        const S_: usize = 31;
        const E_: usize = 1;
        const M_: usize = (S_ - E_ - 10) / 2;
        const N_: usize = S_ - E_ - M_ - 10;

        const S: usize = 1 << S_;
        const E: usize = 1 << E_;
        const M: usize = 1 << M_;
        const N: usize = 1 << N_;

        let mut src = ctx.malloc::<u8>(S);
        let mut dst = ctx.malloc::<u8>(S);

        let dst = dst.as_mut_ptr();
        let src = src.as_ptr();

        let mem = Arr::new_contiguous(&[M, N << 10], BigEndian, E);

        for i in 0..=4 {
            let step = 1 << i;
            let src_ = mem.clone().slice(1, 0, step, (N << 10) >> 4);
            let dst_ = mem.clone().slice(1, 0, 1, (N << 10) >> 4);

            let &[dst_sy, dst_sx] = dst_.strides() else {
                unreachable!()
            };
            let &[src_sy, src_sx] = src_.strides() else {
                unreachable!()
            };
            let dst_sy = dst_sy as i64;
            let dst_sx = dst_sx as i64;
            let src_sy = src_sy as i64;
            let src_sx = src_sx as i64;
            let unit = E as u8;

            let params = cuda::params![dst, dst_sy, dst_sx, src, src_sy, src_sx, unit];
            kernel.launch(
                (M as c_uint, (N >> 4) as c_uint),
                1 << 10,
                params.as_ptr(),
                0,
                None,
            )
        }
    })
}
