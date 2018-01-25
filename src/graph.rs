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

#[derive(Clone, Debug, PartialEq)]
pub enum Slot {
    Zero,
    One,
}

const SLOTS: [Slot; 2] = [Slot::Zero, Slot::One];

pub struct Graph<'a> {
    handle: *const c_void,
    slot: usize,
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
                slot: 0,
                phantom: PhantomData,
            })
    }

    /// Loads an input `tensor`. The type `In` is most likely `::half::f16`.
    ///
    /// Returns the slot in which the tensor was loaded.
    pub fn load_tensor<In>(&mut self, tensor: &[In]) -> Result<Slot, Error> {
        let slot = SLOTS[self.slot..self.slot + 1].as_ptr();
        unsafe {
            api::mvncLoadTensor(
                self.handle,
                tensor.as_ptr() as *const _ as *const c_void,
                (size_of::<In>() * tensor.len()) as c_uint,
                slot as *const c_void,
            )
        }.into_result()
            .map(|()| {
                self.slot = if self.slot == 0 { 1 } else { 0 };
                unsafe { &*slot }.clone()
            })
    }

    /// Gets the next result. The type `Out` is most likely `::half::f16`.
    ///
    /// Returns the output values and the slot from which the result was taken.
    pub fn get_result<Out>(&self) -> Result<(Slot, &[Out]), Error>
    where
        Out: ::std::fmt::Debug,
    {
        let mut p = ptr::null();
        let mut size = 0;
        let mut s = ptr::null();

        unsafe { api::mvncGetResult(self.handle, &mut p, &mut size, &mut s) }
            .into_result()
            .map(|()| {
                if size % size_of::<Out>() as c_uint != 0 {
                    panic!(
                        "Expected multiple of {} bytes for result, got {}",
                        size_of::<Out>(),
                        size
                    );
                }

                unsafe {
                    (
                        (*(s as *const _ as *const Slot)).clone(),
                        slice::from_raw_parts(
                            p as *const _ as *const Out,
                            size as usize / size_of::<Out>(),
                        ),
                    )
                }
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
                    _ => Err(Error::Unknown),
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
