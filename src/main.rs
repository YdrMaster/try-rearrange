use cuda::{Ptx, Symbol};
use ndarray_layout::Endian::BigEndian;
use std::ffi::{c_uint, CString};

type Arr = ndarray_layout::ArrayLayout<4>;

const CODE: &str = include_str!("rearrange.cu");

fn main() {
    // let Some(Symbol::Global(name)) = Symbol::search(CODE).next() else {
    //     return;
    // };

    let name = CString::new("rearrange").unwrap();

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

        let fill_src_name = CString::new("fill_src").unwrap();

        const BLOCK_DIM_EXP: usize = 10;
        const BLOCK_DIM: usize = 1 << BLOCK_DIM_EXP;
        const S_: usize = 31;
        const E_: usize = 4;
        const M_: usize = (S_ - E_ - BLOCK_DIM_EXP) / 2;
        const N_: usize = S_ - E_ - M_ - BLOCK_DIM_EXP;

        const S: usize = 1 << S_;
        const E: usize = 1 << E_;
        const M: usize = 1 << M_;
        const N: usize = 1 << N_;

        let mut src = ctx.malloc::<u8>(S);
        let mut dst = ctx.malloc::<u8>(S);

        let dst = dst.as_mut_ptr();
        let src = src.as_ptr();

        let mem = Arr::new_contiguous(&[M, N << BLOCK_DIM_EXP], BigEndian, E);

        for i in 0..=4 {
            let step = 1 << i;
            let src_ = mem.clone().slice(1, 0, step, (N << BLOCK_DIM_EXP) >> 4);
            let dst_ = mem.clone().slice(1, 0, 1, (N << BLOCK_DIM_EXP) >> 4);

            println!(
                "mem strides: {:?}, mem shape: {:?}",
                mem.strides(),
                mem.shape()
            );

            println!(
                "dst strides: {:?}, dst shape: {:?}",
                dst_.strides(),
                dst_.shape()
            );
            println!(
                "src strides: {:?}, src shape: {:?}",
                src_.strides(),
                src_.shape()
            );

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
                BLOCK_DIM as c_uint,
                params.as_ptr(),
                0,
                None,
            )
        }
    })
}

#[test]
fn test_diff_unit() {
    // let Some(Symbol::Global(name)) = Symbol::search(CODE).next() else {
    //     return;
    // };
    let name = CString::new("rearrange2").unwrap();

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

        // 获取 fill_src kernel
        let fill_src_name = CString::new("fill_src").unwrap();
        let fill_src = module.get_kernel(&fill_src_name);

        const BLOCK_DIM_EXP: usize = 9;
        const BLOCK_DIM: usize = 1 << BLOCK_DIM_EXP;
        const S_: usize = 31;
        const S: usize = 1 << S_;

        let mut src = ctx.malloc::<u8>(S);
        let mut dst = ctx.malloc::<u8>(S);

        // 测试不同的每线程处理数据量
        for items_exp in 0..=5 {
            let items_per_thread = 1 << items_exp;
            println!("\n测试每线程处理 {} 个数据项:", items_per_thread);

            // 初始化源数据
            let block_size = BLOCK_DIM;
            let grid_size =
                (S + (block_size * items_per_thread) - 1) / (block_size * items_per_thread);

            let fill_params = cuda::params![src.as_mut_ptr(), S as u32, items_per_thread as u32];
            fill_src.launch(
                grid_size as c_uint,
                block_size as c_uint,
                fill_params.as_ptr(),
                0,
                None,
            );

            let dst = dst.as_mut_ptr();
            let src = src.as_ptr();

            // 测试不同的单元大小
            for unit_exp in 0..=5 {
                let FAKE_E_: usize = unit_exp + items_exp;
                let M_ = (S_ - FAKE_E_ - BLOCK_DIM_EXP);
                let FAKE_E: usize = 1 << FAKE_E_;
                let M = 1 << M_;

                let mem = Arr::new_contiguous(&[M], BigEndian, FAKE_E);

                let src_ = mem.clone();
                let dst_ = mem.clone();

                let &[dst_sx] = dst_.strides() else {
                    unreachable!()
                };
                let &[src_sx] = src_.strides() else {
                    unreachable!()
                };

                let unit = 1 << unit_exp as u8;
                let dst_sy = 0 as i64;
                let dst_sx = dst_sx as i64;
                let src_sy = 0 as i64;
                let src_sx = src_sx as i64;
                let items_per_thread = items_per_thread as i64;
                println!(
                    "mem strides: {:?}, mem shape: {:?}",
                    mem.strides(),
                    mem.shape()
                );

                let params = cuda::params![
                    dst,
                    dst_sy,
                    dst_sx,
                    src,
                    src_sy,
                    src_sx,
                    unit,
                    items_per_thread
                ];

                println!(
                    "  单元大小: {}, 每线程处理: {}, 启动参数: {:?}",
                    unit, items_per_thread, M as c_uint
                );
                // Calculate grid size based on items_per_thread
                let grid_x = M;

                kernel.launch(
                    grid_x as c_uint,
                    BLOCK_DIM as c_uint,
                    params.as_ptr(),
                    0,
                    None,
                );
            }
        }
    })
}
