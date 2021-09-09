use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Result;
use ash::vk::*;

pub trait AllocationTrait {
    /// Returns the vk::DeviceMemory object that is backing this allocation.
    unsafe fn memory(&self) -> DeviceMemory;

    /// Returns the offset of the allocation on the vk::DeviceMemory. When binding the memory to a buffer or image, this offset needs to be supplied as well.
    fn offset(&self) -> u64;

    /// Returns the size of the allocation
    fn size(&self) -> u64;

    /// Returns a valid mapped pointer if the memory is host visible, otherwise it will return None. The pointer already points to the exact memory region of the suballocation, so no offset needs to be applied.
    fn mapped_ptr(&self) -> Option<NonNull<c_void>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// The allocated resource is stored at an unknown memory location; let the driver decide what’s the best location
    Unknown,
    /// Store the allocation in GPU only accessible memory - typically this is the faster GPU resource and this should be where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers
    CpuToGpu,
    /// Memory useful for CPU readback of data
    GpuToCpu,
}

pub trait AllocationCreateInfoTrait {
    fn new(requirements: MemoryRequirements, location: MemoryLocation, linear: bool) -> Self;
}

pub trait AllocatorTrait {
    type Allocation: AllocationTrait;
    type AllocationCreateInfo: AllocationCreateInfoTrait;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> Result<Self::Allocation>;
    fn free(&self, allocation: Self::Allocation) -> Result<()>;
}
