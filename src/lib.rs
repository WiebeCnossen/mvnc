use std::os::raw::{c_int, c_uint, c_void};
use std::str;

mod api;
pub mod device;
pub mod graph;
pub mod log;

use api::*;

pub use device::Device;
pub use graph::Graph;

fn assert_size(label: &str, expected_size: c_uint, size: c_uint) {
    if expected_size != size {
        panic!(
            "Expected {} bytes for {}, got {}",
            expected_size, label, size
        );
    }
}

fn from_c_string(c_string: &[u8]) -> Result<String, Error> {
    match c_string.iter().position(|&c| c == 0) {
        Some(p) => str::from_utf8(&c_string[0..p])
            .map(|s| s.to_owned())
            .map_err(|_| Error::Unknown),
        None => Err(Error::Unknown),
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    Busy,
    Error,
    OutOfMemory,
    DeviceNotFound,
    InvalidParameters,
    Timeout,
    MvcmdNotFound,
    NoData,
    Gone,
    UnsupportedGraphFile,
    MyriadError,
    Unknown,
}

trait IntoResult {
    fn into_result(self) -> Result<(), Error>;
}

impl IntoResult for c_int {
    fn into_result(self) -> Result<(), Error> {
        match self {
            MVNC_OK => Ok(()),
            MVNC_BUSY => Err(Error::Busy),
            MVNC_ERROR => Err(Error::Error),
            MVNC_OUT_OF_MEMORY => Err(Error::OutOfMemory),
            MVNC_DEVICE_NOT_FOUND => Err(Error::DeviceNotFound),
            MVNC_INVALID_PARAMETERS => Err(Error::InvalidParameters),
            MVNC_TIMEOUT => Err(Error::Timeout),
            MVNC_MVCMD_NOT_FOUND => Err(Error::MvcmdNotFound),
            MVNC_NO_DATA => Err(Error::NoData),
            MVNC_GONE => Err(Error::Gone),
            MVNC_UNSUPPORTED_GRAPH_FILE => Err(Error::UnsupportedGraphFile),
            MVNC_MYRIAD_ERROR => Err(Error::MyriadError),
            _ => Err(Error::Unknown),
        }
    }
}

trait DeviceHandle {
    fn handle(&self) -> *const c_void;
}
