use std::os::raw::c_void;

use api;
use assert_size;
use Error;
use IntoResult;

#[derive(Debug, PartialEq)]
pub enum LogLevel {
    Off,
    Error,
    Verbose,
}

/// Gets the log level
pub fn get_log_level() -> Result<LogLevel, Error> {
    let mut log_level = -1;
    let expected_size = api::SIZEOF_C_INT;
    let mut size = expected_size;

    unsafe {
        api::mvncGetGlobalOption(
            api::MVNC_LOG_LEVEL,
            &mut log_level as *mut _ as *mut c_void,
            &mut size,
        )
    }.into_result()
        .and_then(|()| {
            assert_size("log level", expected_size, size);
            match log_level {
                0 => Ok(LogLevel::Off),
                1 => Ok(LogLevel::Error),
                2 => Ok(LogLevel::Verbose),
                _ => Err(Error::Unknown),
            }
        })
}

/// Sets the log level
pub fn set_log_level(log_level: &LogLevel) -> Result<(), Error> {
    let value = match *log_level {
        LogLevel::Off => 0,
        LogLevel::Error => 1,
        LogLevel::Verbose => 2,
    };

    unsafe {
        api::mvncSetGlobalOption(
            api::MVNC_LOG_LEVEL,
            &value as *const _ as *const c_void,
            api::SIZEOF_C_INT,
        )
    }.into_result()
}
