use std::os::raw::{c_char, c_int, c_uint, c_void};

pub const MVNC_MAX_NAME_SIZE: usize = 28;

pub const MVNC_OK: c_int = 0;
pub const MVNC_BUSY: c_int = -1; // Device is busy, retry later
pub const MVNC_ERROR: c_int = -2; // Error communicating with the device
pub const MVNC_OUT_OF_MEMORY: c_int = -3; // Out of memory
pub const MVNC_DEVICE_NOT_FOUND: c_int = -4; // No device at the given index or name
pub const MVNC_INVALID_PARAMETERS: c_int = -5; // At least one of the given parameters is wrong
pub const MVNC_TIMEOUT: c_int = -6; // Timeout in the communication with the device
pub const MVNC_MVCMD_NOT_FOUND: c_int = -7; // The file to boot Myriad was not found
pub const MVNC_NO_DATA: c_int = -8; // No data to return, call LoadTensor first
pub const MVNC_GONE: c_int = -9; // The graph or device has been closed during the operation
pub const MVNC_UNSUPPORTED_GRAPH_FILE: c_int = -10; // The graph file version is not supported
pub const MVNC_MYRIAD_ERROR: c_int = -11; // An error has been reported by the device, use MVNC_DEBUG_INFO

pub const MVNC_LOG_LEVEL: c_int = 0; // Log level, int, 0 = nothing, 1 = errors, 2 = verbose

/*
pub const MVNC_ITERATIONS: c_int = 0; // Number of iterations per inference, int, normally 1, not for general use
pub const MVNC_NETWORK_THROTTLE: c_int = 1; // Measure temperature once per inference instead of once per layer, int, not for general use
*/
pub const MVNC_DONT_BLOCK: c_int = 2; // LoadTensor will return BUSY instead of blocking, GetResult will return NO_DATA, int
pub const MVNC_TIME_TAKEN: c_int = 1000; // Return time taken for inference (float *)
pub const MVNC_DEBUG_INFO: c_int = 1001; // Return debug info, string

pub const MVNC_TEMP_LIM_NORMAL: c_int = 0; // Undocumented value, assuming it means 'normal'
pub const MVNC_TEMP_LIM_LOWER: c_int = 1; // Temperature for short sleep, float, not for general use
pub const MVNC_TEMP_LIM_HIGHER: c_int = 2; // Temperature for long sleep, float, not for general use

/*
pub const MVNC_BACKOFF_TIME_NORMAL: c_int = 3; // Normal sleep in ms, int, not for general use
pub const MVNC_BACKOFF_TIME_HIGH: c_int = 4; // Short sleep in ms, int, not for general use
pub const MVNC_BACKOFF_TIME_CRITICAL: c_int = 5; // Long sleep in ms, int, not for general use
pub const MVNC_TEMPERATURE_DEBUG: c_int = 6; // Stop on critical temperature, int, not for general use
pub const MVNC_THERMAL_STATS: c_int = 1000; // Return temperatures, float *, not for general use
pub const MVNC_OPTIMISATION_LIST: c_int = 1001; // Return optimisations list, char *, not for general use
*/
pub const MVNC_THERMAL_THROTTLING_LEVEL: c_int = 1002; // 1=TEMP_LIM_LOWER reached, 2=TEMP_LIM_HIGHER reached

#[link(name = "mvnc")]
extern "C" {
    pub fn mvncGetDeviceName(index: c_int, name: *mut c_char, nameSize: c_uint) -> c_int;
    pub fn mvncOpenDevice(name: *const c_char, deviceHandle: &mut *const c_void) -> c_int;
    pub fn mvncCloseDevice(deviceHandle: *const c_void) -> c_int;

    pub fn mvncAllocateGraph(
        deviceHandle: *const c_void,
        graphHandle: &mut *const c_void,
        graphFile: *const c_void,
        graphFileLength: c_uint,
    ) -> c_int;

    pub fn mvncDeallocateGraph(graphHandle: *const c_void) -> c_int;
    pub fn mvncGetGlobalOption(option: c_int, data: *mut c_void, dataLength: *mut c_uint) -> c_int;
    pub fn mvncSetGlobalOption(option: c_int, data: *const c_void, dataLength: c_uint) -> c_int;

    pub fn mvncGetGraphOption(
        graphHandle: *const c_void,
        option: c_int,
        data: *mut c_void,
        dataLength: *mut c_uint,
    ) -> c_int;
    pub fn mvncSetGraphOption(
        graphHandle: *const c_void,
        option: c_int,
        data: *const c_void,
        dataLength: c_uint,
    ) -> c_int;

    pub fn mvncGetDeviceOption(
        deviceHandle: *const c_void,
        option: c_int,
        data: *mut c_void,
        dataLength: *mut c_uint,
    ) -> c_int;

    /*
    // All settable options are marked 'not for general use', hence not included
    pub fn mvncSetDeviceOption(
        deviceHandle: *const c_void,
        option: c_int,
        data: *const c_void,
        dataLength: c_uint,
    ) -> c_int;
*/

    pub fn mvncLoadTensor(
        graphHandle: *const c_void,
        inputTensor: *const c_void,
        inputTensorLength: c_uint,
        userParam: *const c_void,
    ) -> c_int;
    pub fn mvncGetResult(
        graphHandle: *const c_void,
        outputData: &mut *const c_void,
        outputDataLength: *mut c_uint,
        userParam: &mut *const c_void,
    ) -> c_int;
}

pub const SIZEOF_C_FLOAT: c_uint = 4;
pub const SIZEOF_C_INT: c_uint = 4;
