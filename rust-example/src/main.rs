extern crate ocl;
use std::{i32, time::Instant, usize};

#[allow(dead_code)]
fn fourth_elevation_ocl(input: Vec<f32> ,input_size: i32) -> ocl::Result<Vec<f32>> {
    use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

    let kernel_code = include_str!("./fourth_elevation.cl");

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims.
    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Using device: {}", device.name()?);
    println!("Device type: {:?}", device.info(ocl::enums::DeviceInfo::Type)?);
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(kernel_code)
        .build(&context)?;
    let queue = Queue::new(&context, device, None)?;

    // (2) Create a `Buffer`:
    let buffer_in = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(input_size)
        .copy_host_slice(input.as_slice())
        .build()?;

    let buffer_out = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(input_size)
        .fill_val(0.0)
        .build()?;

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::builder()
        .program(&program)
        .name("fourth_elevation")
        .queue(queue.clone())
        .global_work_size(input_size)
        .arg(&buffer_in)
        .arg(&buffer_out)
        .arg(&input_size)
        .build()?;

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(input_size)
            .local_work_size(kernel.default_local_work_size())
            .enq()?;
    }

    // (5) Read results from the device into a vector (`::block` not shown):
    let mut vec = vec![0.0f32; input_size as usize];
    buffer_out.cmd().queue(&queue).offset(0).read(&mut vec).enq()?;

    Ok(vec)
}

fn fourth_elevation_cpu(input: Vec<f32> ,input_size: i32) -> Vec<f32> {
    let input_size = input_size as usize;
    let mut result = vec![0.0f32; input_size];
    for i in 0..input_size {
        result[i] = input[i].powf(4.0);
    }

    result
}

fn main() {
    let relative_error = 0.000001; // 0,0001% relative error
    let input_size: i32 = 1024 * 1024 * 1023;
    let mut input = vec![0.0f32; input_size as usize];
    for i in 0..input_size {
        input[i as usize] = i as f32;
    }
    let start = Instant::now();
    let ocl_result = fourth_elevation_ocl(input.clone(),input_size).unwrap();
    let duration = start.elapsed();
    println!("GPU Took: {:?}", duration);
    let start = Instant::now();
    let cpu_result = fourth_elevation_cpu(input,input_size);
    let duration = start.elapsed();
    println!("CPU Took: {:?}", duration);
    for (index,value) in ocl_result.iter().enumerate() {
        if (cpu_result[index] - value).abs() / value > relative_error {
            println!("Mismatch at index {}: GPU value = {}, CPU value = {}", index, value, cpu_result[index]);
            return;
        }
    }
    println!("All values match!");
}   
