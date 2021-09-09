use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpu_allocator::vulkan::*;

use crate::allocator::{AllocationCreateInfoTrait, AllocationTrait, AllocatorTrait};

impl AllocationTrait for Allocation {
    unsafe fn memory(&self) -> ash::vk::DeviceMemory {
        Allocation::memory(&self)
    }

    fn offset(&self) -> u64 {
        Allocation::offset(&self)
    }

    fn size(&self) -> u64 {
        Allocation::size(&self)
    }

    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        Allocation::mapped_ptr(&self)
    }
}

impl AllocationCreateInfoTrait for AllocationCreateDesc<'static> {
    fn new(
        requirements: ash::vk::MemoryRequirements,
        location: crate::MemoryLocation,
        linear: bool,
    ) -> Self {
        Self {
            name: "egui-winit-ash-integration",
            requirements,
            location: match location {
                crate::MemoryLocation::Unknown => gpu_allocator::MemoryLocation::Unknown,
                crate::MemoryLocation::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
                crate::MemoryLocation::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
                crate::MemoryLocation::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
            },
            linear,
        }
    }
}

impl AllocatorTrait for Arc<Mutex<Allocator>> {
    type Allocation = Allocation;
    type AllocationCreateInfo = AllocationCreateDesc<'static>;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> Result<Self::Allocation> {
        Ok(Allocator::allocate(&mut self.lock().unwrap(), &desc)?)
    }

    fn free(&self, allocation: Self::Allocation) -> Result<()> {
        Ok(Allocator::free(&mut self.lock().unwrap(), allocation)?)
    }
}
