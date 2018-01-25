extern crate half;
extern crate mvnc;
extern crate rand;

use std::fs::File;
use std::io::{self, Read};

use half::{consts, f16};

use mvnc::{Device, Graph};
use mvnc::graph::Blocking;
use mvnc::log;

use rand::{Rng, ThreadRng};

pub fn main() {
    log::set_log_level(&log::LogLevel::Verbose).expect("Setting log level failed");
    for i in 0.. {
        if let Some(device_name) = Device::get_name(i) {
            println!("Device {} = '{}'", i, device_name);
            if let Err(error) = run_mnist(&device_name) {
                println!("{:?}", error);
            }
        } else {
            println!("Finished; # devices = {}", i);
            break;
        }
    }
}

fn read_graph() -> Result<Vec<u8>, io::Error> {
    let mut data = vec![];
    File::open("./examples/mnist.graph")?
        .read_to_end(&mut data)
        .map(|_| data)
}

fn random_input(rng: &mut ThreadRng) -> Vec<f16> {
    (0..768).map(|_| f16::from_f32(rng.gen())).collect()
}

fn run_mnist(device_name: &str) -> Result<(), Error> {
    let mut rng = rand::thread_rng();
    let device = Device::open(device_name)?;

    let data = read_graph()?;
    let mut graph = Graph::allocate(&device, &data)?;

    graph.set_blocking(&Blocking::Block)?;
    println!("Blocking -> {:?}", graph.get_blocking()?);
    for _ in 0..10 {
        exec_block(&mut graph, &mut rng)?;
    }

    graph.set_blocking(&Blocking::DontBlock)?;
    println!("Blocking -> {:?}", graph.get_blocking()?);
    for _ in 0..10 {
        exec_dont_block(&mut graph, &mut rng)?;
    }

    println!(
        "Thermal throttling level = {:?}",
        device.get_thermal_throttling_level()?
    );

    Ok(())
}

fn exec_block(graph: &mut Graph, rng: &mut ThreadRng) -> Result<(), Error> {
    graph.load_tensor(&random_input(rng))?;
    let (id, digit) = graph
        .get_result::<f16>()
        .map(|(id, output)| (id, most_probable_digit(output)))?;
    let time_taken: f32 = graph.get_time_taken()?.iter().cloned().sum();
    print_result(id, digit, time_taken);
    Ok(())
}

fn exec_dont_block(graph: &mut Graph, rng: &mut ThreadRng) -> Result<(), Error> {
    loop {
        match graph.load_tensor(&random_input(rng)) {
            Ok(_) => (),
            Err(mvnc::Error::Busy) => break, // All buffers filled
            Err(e) => return Err(e.into()),
        }
    }

    loop {
        let result = graph
            .get_result::<f16>()
            .map(|(id, output)| (id, most_probable_digit(output)));
        match result {
            Ok((id, digit)) => {
                let time_taken: f32 = graph.get_time_taken()?.iter().cloned().sum();
                print_result(id, digit, time_taken);
            }
            Err(mvnc::Error::Idle) => return Ok(()), // No calculations pending
            Err(mvnc::Error::NoData) => (),          // Calculation not ready
            Err(e) => return Err(e.into()),
        }
    }
}

fn most_probable_digit(output: &[f16]) -> usize {
    let mut max = consts::MIN;
    let mut digit = 0;
    for (i, &prob) in output.iter().enumerate() {
        if prob > max {
            max = prob;
            digit = i;
        }
    }
    digit
}

fn print_result(id: usize, digit: usize, time_taken: f32) {
    println!(
        "Run {:2} in {:.2}ms, most probable digit = {}",
        id, time_taken, digit
    );
}

#[derive(Debug)]
enum Error {
    MvncError(mvnc::Error),
    IoError(io::Error),
}

impl From<mvnc::Error> for Error {
    fn from(error: mvnc::Error) -> Error {
        Error::MvncError(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}
