use std::ffi::CString;
use std::ptr;
use std::os::raw::{c_char, c_int, c_uint, c_void};

use api;
use assert_size;
use from_c_string;

use DeviceHandle;
use Error;
use IntoResult;

#[derive(Debug, PartialEq)]
pub enum ThermalThrottlingLevel {
    Normal,
    TemperatureLimitLowerReached,
    TemperatureLimitHigherReached,
}

pub struct Device {
    handle: *const c_void,
}

impl Device {
    /// Gets the device name at given zero-based `index` or None if there is no device there.
    pub fn get_name(index: usize) -> Option<String> {
        let mut buf = [0; api::MVNC_MAX_NAME_SIZE];
        unsafe {
            api::mvncGetDeviceName(
                index as c_int,
                buf.as_mut_ptr() as *mut _ as *mut c_char,
                buf.len() as c_uint,
            )
        }.into_result()
            .and_then(|()| from_c_string(&buf))
            .ok()
    }

    /// Opens the device with the given `name`.
    pub fn open(name: &str) -> Result<Device, Error> {
        let cstr = CString::new(name).unwrap();
        let mut handle = ptr::null::<c_void>();
        unsafe { api::mvncOpenDevice(cstr.as_ptr(), &mut handle).into_result() }
            .map(|()| Device { handle })
    }

    /// Gets the thermal throttling level.
    pub fn get_thermal_throttling_level(&self) -> Result<ThermalThrottlingLevel, Error> {
        let mut thermal_throttling_level = -1;
        let expected_size = api::SIZEOF_C_INT;
        let mut size = expected_size;

        unsafe {
            api::mvncGetDeviceOption(
                self.handle,
                api::MVNC_THERMAL_THROTTLING_LEVEL,
                &mut thermal_throttling_level as *mut _ as *mut c_void,
                &mut size,
            )
        }.into_result()
            .and_then(|()| {
                assert_size("throttling level", expected_size, size);
                match thermal_throttling_level {
                    api::MVNC_TEMP_LIM_NORMAL => Ok(ThermalThrottlingLevel::Normal),
                    api::MVNC_TEMP_LIM_LOWER => {
                        Ok(ThermalThrottlingLevel::TemperatureLimitLowerReached)
                    }
                    api::MVNC_TEMP_LIM_HIGHER => {
                        Ok(ThermalThrottlingLevel::TemperatureLimitHigherReached)
                    }
                    _ => Err(Error::ApiError),
                }
            })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Err(e) = unsafe { api::mvncCloseDevice(self.handle) }.into_result() {
            eprintln!("::mvnc::device::Device::drop: Err({:?})", e);
        }
    }
}

impl DeviceHandle for Device {
    fn handle(&self) -> *const c_void {
        self.handle
    }
}
