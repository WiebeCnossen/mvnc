# mvnc

Wrapper around the Movidius Neural Computing stick C API.

## Version history

Version | Description
------- | -----------------------------------------------
0.2.1   | Moved repository to github
0.2.0   | Complete crate including Slot indicator
0.1.3   | Implemented `graph` module
0.1.2   | Implemented `device` module
0.1.1   | Added internal `IntoResult` trait and readme.md
0.1.0   | Implemented `log` module

## Build instructions

The `libmvnc.so` library from the Movidius™ Neural Compute SDK must be present.

## Example

This example can be found at https://github.com/WiebeCnossen/mvnc/blob/master/examples/mnist.rs

```
extern crate half;
extern crate mvnc;
extern crate rand;

use std::fs::File;
use std::io;
use std::io::Read;

use half::f16;

use mvnc::{Device, Error, Graph};
use mvnc::graph::Blocking;
use mvnc::log;

use rand::Rng;

pub fn main() {
    log::set_log_level(&log::LogLevel::Verbose).expect("Setting log level failed");
    for _ in 0..2 {
        for i in 0.. {
            if let Some(device_name) = Device::get_name(i) {
                println!("Device {} = '{}'", i, device_name);
                if let Err(error) = run_mnist(&device_name) {
                    println!("Bummer: {:?}", error);
                }
            } else {
                break;
            }
            log::set_log_level(&log::LogLevel::Off).expect("Setting log level failed");
        }
    }
}

fn graph() -> Result<Vec<u8>, io::Error> {
    let mut data = vec![];
    File::open("./examples/mnist.graph")?
        .read_to_end(&mut data)
        .map(|_| data)
}

fn run_mnist(device_name: &str) -> Result<(), Error> {
    let device = Device::open(device_name)?;
    println!(
        "Thermal throttling level = {:?}",
        device.get_thermal_throttling_level()?
    );
    let data = graph().unwrap();
    println!("Data = {} bytes", data.len());
    let mut graph = Graph::allocate(&device, &data)?;

    println!("Blocking = {:?}", graph.get_blocking()?);
    graph.set_blocking(&Blocking::DontBlock)?;

    for i in 0..5 {
        println!("---------- iteration {} ----------", i);
        loop {
            let mut rng = rand::thread_rng();
            let inputs: Vec<_> = (0..768).map(|_| f16::from_f32(rng.gen())).collect();
            match graph.load_tensor(&inputs) {
                Ok(slot) => println!("Tensor loaded into slot {:?}", slot),
                Err(Error::Busy) => {
                    println!("Busy device, let's get results...");
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        loop {
            match graph.get_result::<f16>() {
                Ok((slot, probs)) => {
                    let time_taken = graph.get_time_taken()?.iter().cloned().sum::<f32>();
                    let debug_info = graph.get_debug_info()?;
                    println!(
                        "Slot {:?}, o[8] = {} in {:.2}ms, ({})",
                        slot, probs[8], time_taken, debug_info
                    );
                }
                Err(Error::NoData) => {
                    println!("Empty device: iteration {} done!", i);
                    break;
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}
```
