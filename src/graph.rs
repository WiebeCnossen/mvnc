use std::marker::PhantomData;
use std::mem::size_of;
use std::os::raw::{c_uint, c_void};
use std::ptr;
use std::slice;

use DeviceHandle;
use Error;
use IntoResult;
use api;
use assert_size;
use from_c_string;
use device::Device;

#[derive(Debug, PartialEq)]
pub enum Blocking {
    Block,
    DontBlock,
}

const ID_COUNT: usize = 3;

pub struct Graph<'a> {
    handle: *const c_void,
    next_id: usize,
    ids: [usize; ID_COUNT],
    result_id: usize,
    phantom: PhantomData<(&'a Device)>,
}

impl<'a> Graph<'a> {
    /// Allocates a graph on an opened `device`.
    pub fn allocate<G>(device: &'a Device, graph: &[G]) -> Result<Self, Error> {
        let mut handle = ptr::null();
        unsafe {
            api::mvncAllocateGraph(
                device.handle(),
                &mut handle,
                graph.as_ptr() as *const _ as *const c_void,
                (size_of::<G>() * graph.len()) as c_uint,
            )
        }.into_result()
            .map(|()| Self {
                handle,
                next_id: 1,
                ids: [0; ID_COUNT],
                result_id: 0,
                phantom: PhantomData,
            })
    }

    /// Loads an input `tensor`. The type `In` is most likely `::half::f16`.
    ///
    /// Returns the id of the calculation.
    pub fn load_tensor<In>(&mut self, tensor: &[In]) -> Result<usize, Error> {
        let i = self.next_id % self.ids.len();
        self.ids[i] = self.next_id;
        unsafe {
            api::mvncLoadTensor(
                self.handle,
                tensor.as_ptr() as *const _ as *const c_void,
                (size_of::<In>() * tensor.len()) as c_uint,
                self.ids[i..].as_ptr() as *const c_void,
            )
        }.into_result()
            .map(|()| {
                let id = self.next_id;
                self.next_id += 1;
                id
            })
    }

    /// Gets the next result. The type `Out` is most likely `::half::f16`.
    ///
    /// Returns the id of the calculation and its output.
    ///
    /// Noteworthy errors:
    /// * `Idle`: there are no pending calculations
    /// * `NoData`: pending calculation is not ready,
    ///    occurs only if `Blocking::DontBlock` is set.
    /// * `ApiError`: size of result data is not a multiple of the size of `Out`.
    pub fn get_result<Out>(&mut self) -> Result<(usize, &[Out]), Error> {
        let mut result_ptr = ptr::null();
        let mut size = 0;
        let mut s = ptr::null();

        if self.result_id + 1 == self.next_id {
            return Err(Error::Idle);
        }

        unsafe { api::mvncGetResult(self.handle, &mut result_ptr, &mut size, &mut s) }
            .into_result()
            .and_then(|()| {
                let result_size = size as usize;
                if result_size % size_of::<Out>() != 0 {
                    return Err(Error::ApiError);
                }

                self.result_id = unsafe { *(s as *const _ as *const usize) };
                Ok((self.result_id, unsafe {
                    slice::from_raw_parts(
                        result_ptr as *const _ as *const Out,
                        result_size / size_of::<Out>(),
                    )
                }))
            })
    }

    /// Gets the setting for blocking behaviour.
    pub fn get_blocking(&self) -> Result<Blocking, Error> {
        let mut blocking = -1;
        let expected_size = api::SIZEOF_C_INT;
        let mut size = expected_size;

        unsafe {
            api::mvncGetGraphOption(
                self.handle,
                api::MVNC_DONT_BLOCK,
                &mut blocking as *mut _ as *mut c_void,
                &mut size,
            )
        }.into_result()
            .and_then(|()| {
                assert_size("graph blocking", expected_size, size);
                match blocking {
                    0 => Ok(Blocking::Block),
                    1 => Ok(Blocking::DontBlock),
                    _ => Err(Error::ApiError),
                }
            })
    }

    /// Sets the setting for blocking behaviour.
    pub fn set_blocking(&self, blocking: &Blocking) -> Result<(), Error> {
        let value = match *blocking {
            Blocking::Block => 0,
            Blocking::DontBlock => 1,
        };

        unsafe {
            api::mvncSetGraphOption(
                self.handle,
                api::MVNC_DONT_BLOCK,
                &value as *const _ as *const c_void,
                api::SIZEOF_C_INT,
            )
        }.into_result()
    }

    /// Gets the times taken per stage.
    pub fn get_time_taken(&self) -> Result<&[f32], Error> {
        let mut p = ptr::null();
        let mut size = 0;

        unsafe {
            api::mvncGetGraphOption(
                self.handle,
                api::MVNC_TIME_TAKEN,
                (&mut p) as *mut _ as *mut c_void,
                &mut size,
            )
        }.into_result()
            .map(|()| unsafe { slice::from_raw_parts(p, (size / api::SIZEOF_C_FLOAT) as usize) })
    }

    /// Gets debug info which will be present after some `Error::MyriadError`.
    pub fn get_debug_info(&self) -> Result<String, Error> {
        let mut p = ptr::null();
        let mut size = 0;
        unsafe {
            api::mvncGetGraphOption(
                self.handle,
                api::MVNC_DEBUG_INFO,
                &mut p as *mut _ as *mut c_void,
                &mut size,
            )
        }.into_result()
            .and_then(|()| from_c_string(unsafe { slice::from_raw_parts(p, size as usize) }))
    }
}

impl<'a> Drop for Graph<'a> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { api::mvncDeallocateGraph(self.handle) }.into_result() {
            eprintln!("::mvnc::graph::Graph::drop: Err({:?})", e);
        }
    }
}
