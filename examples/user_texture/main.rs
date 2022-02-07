use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::os::raw::c_char;
#[cfg(debug_assertions)]
use std::os::raw::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{collections::HashSet, u64};

use anyhow::Result;
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::{vk, Device, Entry, Instance};
use ash_window::{create_surface, enumerate_required_extensions};
use cgmath::{Deg, Matrix4, Point3, Vector3};
use crevice::std140::{AsStd140, Std140};
use gpu_allocator::vulkan::*;
use memoffset::offset_of;
#[cfg(debug_assertions)]
use vk::DebugUtilsMessengerEXT;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;
const VALIDATION: &[&str] = &["VK_LAYER_KHRONOS_validation"];

const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

const VERTEX_SHADER_PASS: &'static str = "examples/user_texture/shaders/spv/vert.spv";
const FRAGMENT_SHADER_PASS: &'static str = "examples/user_texture/shaders/spv/frag.spv";

const MODEL_PATH: &'static str = "examples/user_texture/assets/monkey.obj";
const IMAGE_PATH: &'static str = "examples/user_texture/assets/Mandrill.bmp";

const SCENE_IMAGE_WIDTH: u32 = 640;
const SCENE_IMAGE_HEIGHT: u32 = 420;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// convert vk string to String
fn vk_to_string(raw_char_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_char_array.as_ptr();
        CStr::from_ptr(pointer)
    };
    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

// Add perspective method
trait Matrix4Ext {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspecf32: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32>;
}
impl Matrix4Ext for Matrix4<f32> {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        use cgmath::{Angle, Rad};
        let f: Rad<f32> = fovy.into();
        let f = f / 2.0;
        let f = Rad::cot(f);
        Matrix4::<f32>::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            -f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            (near * far) / (near - far),
            0.0,
        )
    }
}

// Vertex struct
#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, position) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, normal) as u32)
                .build(),
        ]
    }
}

// Uniform Buffer object
#[derive(AsStd140)]
struct UniformBufferObject {
    light: mint::Vector3<f32>,
    model: mint::ColumnMatrix4<f32>,
    view: mint::ColumnMatrix4<f32>,
    proj: mint::ColumnMatrix4<f32>,
}

// main app
struct App {
    width: u32,
    height: u32,
    window: Window,
    _entry: Entry,
    instance: Instance,
    #[cfg(debug_assertions)]
    debug_utils_loader: DebugUtils,
    #[cfg(debug_assertions)]
    debug_callback: DebugUtilsMessengerEXT,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue_index: u32,
    present_queue_index: u32,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_command_pool: vk::CommandPool,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    scene_render_pass: vk::RenderPass,
    color_images: Vec<vk::Image>,
    color_image_allocations: Vec<Allocation>,
    color_image_views: Vec<vk::ImageView>,
    depth_images: Vec<vk::Image>,
    depth_image_allocations: Vec<Allocation>,
    depth_image_views: Vec<vk::ImageView>,
    scene_framebuffers: Vec<vk::Framebuffer>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,
    fences: Vec<vk::Fence>,
    current_frame: usize,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    command_buffers: Vec<vk::CommandBuffer>,

    egui_integration: ManuallyDrop<egui_winit_ash_integration::Integration<Arc<Mutex<Allocator>>>>,
    show_user_texture_window: bool,
    show_scene_window: bool,
    prev_pointer_pos: Option<egui::Pos2>,
    rotation_x: f32,
    rotation_y: f32,
    clear_color: [f32; 4],
    light_position: Vector3<f32>,

    // user texture
    image: vk::Image,
    image_allocation: Option<Allocation>,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    image_texture_id: egui::TextureId,
    scene_texture_ids: Vec<egui::TextureId>,
}
impl App {
    fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let (width, height) = (800, 600);
        let title = "Test";

        // Create Window
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(width, height))
            .with_resizable(true)
            .build(event_loop)?;

        // Create Entry
        let entry = Entry::linked();

        // Create Instance
        let instance = {
            // App info
            let app_name = CString::new(title)?;
            let engine_name = CString::new("Vulkan Engine")?;
            let app_info = vk::ApplicationInfo::builder()
                .api_version(vk::make_api_version(0, 1, 2, 0))
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .application_name(&app_name)
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(&engine_name);

            // Get extensions for creating Surface
            let extension_names = enumerate_required_extensions(&window)?;
            let mut extension_names = extension_names
                .iter()
                .map(|name| name.as_ptr())
                .collect::<Vec<_>>();
            if ENABLE_VALIDATION_LAYERS {
                extension_names.push(DebugUtils::name().as_ptr());
            }

            // layer for validation
            let enabled_layer_names = VALIDATION
                .iter()
                .map(|layer_name| CString::new(*layer_name).unwrap())
                .collect::<Vec<_>>();
            let enabled_layer_names = enabled_layer_names
                .iter()
                .map(|layer_name| layer_name.as_ptr())
                .collect::<Vec<_>>();

            // instance create info
            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&extension_names);
            let create_info = if ENABLE_VALIDATION_LAYERS {
                create_info.enabled_layer_names(&enabled_layer_names)
            } else {
                create_info
            };

            // crate instance
            unsafe { entry.create_instance(&create_info, None)? }
        };

        #[cfg(debug_assertions)]
        let (debug_utils_loader, debug_callback) = {
            // callback function
            unsafe extern "system" fn callback(
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT,
                p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
                _p_user_data: *mut c_void,
            ) -> vk::Bool32 {
                let severity = match message_severity {
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
                    vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
                    _ => "[Unknown]",
                };
                let types = match message_type {
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
                    vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
                    _ => "[Unknown]",
                };
                let message = CStr::from_ptr((*p_callback_data).p_message);
                println!("[Debug]{}{}{:?}", severity, types, message);

                vk::FALSE
            }
            let debug_utils_messenger_create_info_ext =
                vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                            // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    )
                    .pfn_user_callback(Some(callback));
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_utils_messenger_create_info_ext, None)?
            };
            (debug_utils_loader, debug_callback)
        };

        // Create Surface
        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe { create_surface(&entry, &instance, &window, None)? };

        // Select Physical Device
        let (physical_device, graphics_queue_index, present_queue_index) = {
            // filter physical device
            let physical_devices = unsafe { instance.enumerate_physical_devices()? };
            let mut physical_devices = physical_devices.into_iter().filter_map(|physical_device| {
                // Check if the required queue family is supported.
                // Check if graphics and present are supported.
                // In some cases, both may have the same index.
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };
                let mut graphics_queue_index = None;
                let mut present_queue_index = None;
                for (i, queue_family) in queue_families.iter().enumerate() {
                    if queue_family.queue_count > 0
                        && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    {
                        graphics_queue_index = Some(i as u32);
                    }

                    let is_present_support = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap()
                    };
                    if queue_family.queue_count > 0 && is_present_support {
                        present_queue_index = Some(i as u32);
                    }

                    if graphics_queue_index.is_some() && present_queue_index.is_some() {
                        break;
                    }
                }
                let is_queue_families_supported =
                    graphics_queue_index.is_some() && present_queue_index.is_some();

                // Check if the extensions specified in DEVICE_EXTENSIONS are supported.
                let is_device_extension_supported = {
                    let available_extensions = unsafe {
                        instance
                            .enumerate_device_extension_properties(physical_device)
                            .unwrap()
                    };
                    let mut available_extension_names = vec![];
                    for extension in available_extensions.iter() {
                        let extension_name = vk_to_string(&extension.extension_name);
                        available_extension_names.push(extension_name);
                    }
                    let mut required_extensions = HashSet::new();
                    for extension in DEVICE_EXTENSIONS.iter() {
                        required_extensions.insert(extension.to_string());
                    }
                    for extension in available_extension_names.iter() {
                        required_extensions.remove(extension);
                    }
                    required_extensions.is_empty()
                };

                // Check if Swapchain is supported.
                let is_swapchain_supported = if is_device_extension_supported {
                    let formats = unsafe {
                        surface_loader
                            .get_physical_device_surface_formats(physical_device, surface)
                            .unwrap()
                    };
                    let present_modes = unsafe {
                        surface_loader
                            .get_physical_device_surface_present_modes(physical_device, surface)
                            .unwrap()
                    };
                    !formats.is_empty() && !present_modes.is_empty()
                } else {
                    false
                };

                if is_queue_families_supported
                    && is_device_extension_supported
                    && is_swapchain_supported
                {
                    Some((
                        physical_device,
                        graphics_queue_index.unwrap(),
                        present_queue_index.unwrap(),
                    ))
                } else {
                    None
                }
            });

            // Select the first physical device that satisfies the conditions.
            physical_devices.next().unwrap()
        };

        // Create Device
        let device = {
            let mut unique_queue_families = HashSet::new();
            unique_queue_families.insert(graphics_queue_index);
            unique_queue_families.insert(present_queue_index);

            let queue_priorities = [1.0];
            let mut queue_create_infos = vec![];
            for &queue_family in unique_queue_families.iter() {
                let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family)
                    .queue_priorities(&queue_priorities)
                    .build();
                queue_create_infos.push(queue_create_info);
            }

            let enabled_extension_names = [Swapchain::name().as_ptr()];

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(queue_create_infos.as_slice())
                .enabled_extension_names(&enabled_extension_names);

            unsafe { instance.create_device(physical_device, &device_create_info, None)? }
        };

        // Create Queues
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_index, 0) };

        // Create Graphics Command Pool
        let graphics_command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_queue_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&command_pool_create_info, None)? }
        };

        // Create Swapchain
        let swapchain_loader = Swapchain::new(&instance, &device);
        let (swapchain, format, extent) = {
            let capabilities = unsafe {
                surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
            };
            let formats = unsafe {
                surface_loader
                    .get_physical_device_surface_formats(physical_device, surface)
                    .unwrap()
            };
            let present_modes = unsafe {
                surface_loader
                    .get_physical_device_surface_present_modes(physical_device, surface)
                    .unwrap()
            };

            let format = formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM
                })
                .unwrap_or(&formats[0])
                .clone();
            let present_mode = present_modes
                .into_iter()
                .find(|&p| p == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let extent = {
                if capabilities.current_extent.width != u32::max_value() {
                    capabilities.current_extent
                } else {
                    vk::Extent2D {
                        width: width
                            .max(capabilities.min_image_extent.width)
                            .min(capabilities.max_image_extent.width),
                        height: height
                            .max(capabilities.min_image_extent.height)
                            .min(capabilities.max_image_extent.height),
                    }
                }
            };

            let image_count = capabilities.min_image_count + 1;
            let image_count = if capabilities.max_image_count != 0 {
                image_count.min(capabilities.max_image_count)
            } else {
                image_count
            };

            let (image_sharing_mode, queue_family_indices) =
                if graphics_queue_index != present_queue_index {
                    (
                        vk::SharingMode::EXCLUSIVE,
                        vec![graphics_queue_index, present_queue_index],
                    )
                } else {
                    (vk::SharingMode::EXCLUSIVE, vec![])
                };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(queue_family_indices.as_slice())
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain =
                unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
            (swapchain, format, extent)
        };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        // Prepare gpu-allocator's Allocator
        let mut allocator = {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
            })?
        };

        // Create scene RenderPass
        let scene_render_pass = {
            // Attachments
            let attachments = [
                vk::AttachmentDescription::builder()
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .build(),
                vk::AttachmentDescription::builder()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .build(),
            ];
            // color reference
            let color_reference = [vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()];
            // depth reference
            let depth_reference = vk::AttachmentReference::builder()
                .attachment(1)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            // subpass descriptionを作成
            let subpasses = [vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_reference)
                .depth_stencil_attachment(&depth_reference)
                .build()];
            // create render pass
            let render_pass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            unsafe { device.create_render_pass(&render_pass_create_info, None)? }
        };

        // Create scene images
        let (
            color_images,
            color_image_allocations,
            color_image_views,
            depth_images,
            depth_image_allocations,
            depth_image_views,
            scene_framebuffers,
        ) = {
            let mut color_images = vec![];
            let mut color_image_allocations = vec![];
            let mut color_image_views = vec![];
            let mut depth_images = vec![];
            let mut depth_image_allocations = vec![];
            let mut depth_image_views = vec![];
            let mut scene_framebuffers = vec![];

            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let mut attachments = vec![];

                let color_image_create_info = vk::ImageCreateInfo::builder()
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .mip_levels(1)
                    .array_layers(1)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: SCENE_IMAGE_WIDTH,
                        height: SCENE_IMAGE_HEIGHT,
                        depth: 1,
                    });
                let color_image = unsafe { device.create_image(&color_image_create_info, None)? };
                let color_image_requirements =
                    unsafe { device.get_image_memory_requirements(color_image) };
                let color_image_allocation = allocator.allocate(&AllocationCreateDesc {
                    name: "color image",
                    requirements: color_image_requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                })?;
                unsafe {
                    device.bind_image_memory(
                        color_image,
                        color_image_allocation.memory(),
                        color_image_allocation.offset(),
                    )?;
                }
                color_images.push(color_image);
                color_image_allocations.push(color_image_allocation);
                let color_attachment = unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(color_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R16G16B16A16_SFLOAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )?
                };
                attachments.push(color_attachment);
                color_image_views.push(color_attachment);

                let depth_image_create_info = vk::ImageCreateInfo::builder()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .mip_levels(1)
                    .array_layers(1)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: SCENE_IMAGE_WIDTH,
                        height: SCENE_IMAGE_HEIGHT,
                        depth: 1,
                    });
                let depth_image = unsafe { device.create_image(&depth_image_create_info, None)? };
                let depth_image_requirements =
                    unsafe { device.get_image_memory_requirements(depth_image) };
                let depth_image_allocation = allocator.allocate(&AllocationCreateDesc {
                    name: "depth image",
                    requirements: depth_image_requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                })?;
                unsafe {
                    device.bind_image_memory(
                        depth_image,
                        depth_image_allocation.memory(),
                        depth_image_allocation.offset(),
                    )?;
                }
                depth_images.push(depth_image);
                depth_image_allocations.push(depth_image_allocation);
                let depth_attachment = unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(depth_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )?
                };
                attachments.push(depth_attachment);
                depth_image_views.push(depth_attachment);
                scene_framebuffers.push(unsafe {
                    device.create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(scene_render_pass)
                            .attachments(attachments.as_slice())
                            .width(SCENE_IMAGE_WIDTH)
                            .height(SCENE_IMAGE_HEIGHT)
                            .layers(1),
                        None,
                    )?
                });
            }

            (
                color_images,
                color_image_allocations,
                color_image_views,
                depth_images,
                depth_image_allocations,
                depth_image_views,
                scene_framebuffers,
            )
        };

        // Prepare UniformBuffer
        let (uniform_buffers, uniform_buffer_allocations) = {
            (0..swapchain_images.len())
                .map(|_| {
                    // Calculate size
                    let buffer_size = std::mem::size_of::<Std140UniformBufferObject>() as u64;
                    // Reserve buffer
                    let buffer = unsafe {
                        device
                            .create_buffer(
                                &vk::BufferCreateInfo::builder()
                                    .size(buffer_size)
                                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
                                None,
                            )
                            .unwrap()
                    };
                    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
                    let buffer_allocation = allocator
                        .allocate(&AllocationCreateDesc {
                            name: "uniform buffer",
                            requirements,
                            location: gpu_allocator::MemoryLocation::CpuToGpu,
                            linear: true,
                        })
                        .unwrap();
                    unsafe {
                        device
                            .bind_buffer_memory(
                                buffer,
                                buffer_allocation.memory(),
                                buffer_allocation.offset(),
                            )
                            .unwrap();
                    }
                    (buffer, buffer_allocation)
                })
                .unzip::<_, _, Vec<_>, Vec<_>>()
        };
        // Create DescriptorPool
        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(swapchain_images.len() as u32)
                .build()];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(swapchain_images.len() as u32)
                .pool_sizes(&pool_sizes);
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? }
        };
        // Create Descriptor Set Layout Bindings
        let descriptor_set_layout_bindings = vec![vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .binding(0)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build()];
        // Create Descriptor Set Layout
        let descriptor_set_layouts = (0..swapchain_images.len())
            .map(|_| {
                let descriptor_set_layout_create_info =
                    vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(descriptor_set_layout_bindings.as_slice());
                unsafe {
                    device
                        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>();
        // Cerate Descriptor Sets
        let descriptor_sets = {
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(descriptor_set_layouts.as_slice());
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
        }?;
        // Update Descriptor sets
        for i in 0..swapchain_images.len() {
            let descriptor_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(std::mem::size_of::<Std140UniformBufferObject>() as u64)
                .build()];
            let write_descriptor_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .build()];
            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }

        // Setup Graphics Pipeline
        let (graphics_pipeline, pipeline_layout) = {
            // load shaders
            let vertex_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new(VERTEX_SHADER_PASS))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe { device.create_shader_module(&shader_module_create_info, None)? }
            };
            let fragment_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new(FRAGMENT_SHADER_PASS))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe { device.create_shader_module(&shader_module_create_info, None)? }
            };
            // name main function
            let main_function_name = CString::new("main").unwrap();
            // prepare shader stages info
            let pipeline_shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name)
                    .build(),
            ];
            // Pipeline Layout
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts),
                    None,
                )?
            };
            // VertexInputBinding
            let vertex_input_binding = Vertex::get_binding_descriptions();
            // Vertex Input Attribute Descriptions
            let vertex_input_attribute = Vertex::get_attribute_descriptions();
            // input_assembly info
            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            // viewport info
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            // rasterization_info
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            // stencil op
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            // depth stencil op
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            // color blend attachments

            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B, // | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .build()];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);
            // dynamic state
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            // Vertex input state create info
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_input_attribute)
                .vertex_binding_descriptions(&vertex_input_binding);
            // multi sample info
            let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            // pipeline create info
            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&pipeline_shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_info)
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(scene_render_pass)
                .subpass(0)
                .build()];
            // Graphics Pipeline
            let graphics_pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &pipeline_create_info,
                        None,
                    )
                    .unwrap()[0]
            };
            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }

            (graphics_pipeline, pipeline_layout)
        };

        // vertices
        let vertices = {
            let model_obj = tobj::load_obj(
                MODEL_PATH,
                &tobj::LoadOptions {
                    single_index: true,
                    triangulate: true,
                    ..Default::default()
                },
            )?;
            let mut vertices = vec![];
            let (models, _) = model_obj;
            for m in models.iter() {
                let mesh = &m.mesh;
                for &i in mesh.indices.iter() {
                    let i = i as usize;
                    let vertex = Vertex {
                        position: Vector3::new(
                            mesh.positions[3 * i],
                            mesh.positions[3 * i + 1],
                            mesh.positions[3 * i + 2],
                        ),
                        normal: Vector3::new(
                            mesh.normals[3 * i],
                            mesh.normals[3 * i + 1],
                            mesh.normals[3 * i + 2],
                        ),
                    };
                    vertices.push(vertex);
                }
            }
            vertices
        };
        let vertex_buffer_size = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;

        // temporary CPU to GPU buffer
        let temporary_vertex_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(vertex_buffer_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )?
        };
        let requirements =
            unsafe { device.get_buffer_memory_requirements(temporary_vertex_buffer) };
        let temporary_vertex_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Temporary Vertex Buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
        })?;
        unsafe {
            device.bind_buffer_memory(
                temporary_vertex_buffer,
                temporary_vertex_buffer_allocation.memory(),
                temporary_vertex_buffer_allocation.offset(),
            )?;
        }
        unsafe {
            let ptr = temporary_vertex_buffer_allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut Vertex;
            ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        }

        // vertex buffer
        let vertex_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(vertex_buffer_size)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                    ),
                None,
            )?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(vertex_buffer) };
        let vertex_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Vertex Buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
        })?;
        unsafe {
            device.bind_buffer_memory(
                vertex_buffer,
                vertex_buffer_allocation.memory(),
                vertex_buffer_allocation.offset(),
            )?;
        }

        // copy vertices data from temporary buffer to vertex buffer
        {
            // create onetime command
            let copy_cmd = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(graphics_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?
            }[0];
            unsafe {
                device.begin_command_buffer(
                    copy_cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                device.cmd_copy_buffer(
                    copy_cmd,
                    temporary_vertex_buffer,
                    vertex_buffer,
                    &[vk::BufferCopy::builder()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(vertex_buffer_size)
                        .build()],
                );
                device.end_command_buffer(copy_cmd)?;

                device.queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[copy_cmd])
                        .build()],
                    vk::Fence::null(),
                )?;
                device.queue_wait_idle(graphics_queue)?;

                device.free_command_buffers(graphics_command_pool, &[copy_cmd]);
            }

            // destroy temporary buffer
            allocator.free(temporary_vertex_buffer_allocation)?;
            unsafe {
                device.destroy_buffer(temporary_vertex_buffer, None);
            }
        }

        // Load User Texture image
        let (image, image_allocation, image_view) = {
            let image_object = image::open(&Path::new(IMAGE_PATH))?;
            let (image_width, image_height) = (image_object.width(), image_object.height());
            let image_data = image_object.to_rgba8().into_raw();

            let image_size =
                (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as u64;

            // Create Staging buffer
            let staging_buffer = unsafe {
                device.create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(image_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )?
            };
            let staging_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(staging_buffer) };
            let staging_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
                name: "User Texture Staging Buffer",
                requirements: staging_buffer_requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            })?;
            unsafe {
                device.bind_buffer_memory(
                    staging_buffer,
                    staging_buffer_allocation.memory(),
                    staging_buffer_allocation.offset(),
                )?
            }
            // Map staging buffer
            unsafe {
                let mapped_memory =
                    staging_buffer_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
                mapped_memory.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());
            }

            let format = vk::Format::R8G8B8A8_UNORM;

            // Create Image
            let image = unsafe {
                device.create_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width: image_width,
                            height: image_height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
                    None,
                )?
            };
            let image_requirements = unsafe { device.get_image_memory_requirements(image) };
            let image_allocation = allocator.allocate(&AllocationCreateDesc {
                name: "User Texture Image",
                requirements: image_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            })?;
            unsafe {
                device.bind_image_memory(
                    image,
                    image_allocation.memory(),
                    image_allocation.offset(),
                )?
            };

            // Copy data from buffer to image
            unsafe {
                let command = device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(graphics_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0];

                // Begin command
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                // Change image layout to transfer dst optimal
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                // Copy data from buffer to image
                device.cmd_copy_buffer_to_image(
                    command,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::builder()
                        .image_subresource(
                            vk::ImageSubresourceLayers::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .image_extent(vk::Extent3D {
                            width: image_width,
                            height: image_height,
                            depth: 1,
                        })
                        .buffer_offset(0)
                        .buffer_image_height(0)
                        .buffer_row_length(0)
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .build()],
                );

                // Change image layout to shader read only optimal
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                // End command
                device.end_command_buffer(command)?;

                // Submit command
                device.queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[command])
                        .build()],
                    vk::Fence::null(),
                )?;
                device.queue_wait_idle(graphics_queue)?;
            }

            // Delete staging buffer
            allocator.free(staging_buffer_allocation)?;
            unsafe {
                device.destroy_buffer(staging_buffer, None);
            }

            // Create Image View
            let image_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .image(image)
                        .format(format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )?
            };

            (image, image_allocation, image_view)
        };
        // Create User Texture image sampler
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }?;

        // Create sync objects
        let (fences, image_available_semaphores, render_finished_semaphores) = unsafe {
            let mut fences = vec![];
            let mut image_available_semaphores = vec![];
            let mut render_finished_semaphores = vec![];
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                fences.push(device.create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
                image_available_semaphores
                    .push(device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?);
                render_finished_semaphores
                    .push(device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?);
            }
            (
                fences,
                image_available_semaphores,
                render_finished_semaphores,
            )
        };

        // create command_buffers
        let command_buffers = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(swapchain_images.len() as u32),
            )?
        };

        // Wrap the allocator with ArcMutex to share the allocator between integration and App.
        let allocator = Arc::new(Mutex::new(allocator));

        // #### egui ##########################################################################
        // create integration object
        // Note: ManuallyDrop is required to drop the allocator to shut it down successfully.
        let mut egui_integration = ManuallyDrop::new(egui_winit_ash_integration::Integration::new(
            width,
            height,
            window.scale_factor(),
            egui::FontDefinitions::default(),
            egui::Style::default(),
            device.clone(),
            Arc::clone(&allocator),
            swapchain_loader.clone(),
            swapchain.clone(),
            format.clone(),
        ));

        // Register user texture
        let image_texture_id =
            egui_integration.register_user_texture(image_view.clone(), sampler.clone());
        let scene_texture_ids = color_image_views
            .iter()
            .map(|image_view| {
                egui_integration.register_user_texture(image_view.clone(), sampler.clone())
            })
            .collect::<Vec<_>>();
        // #### egui ##########################################################################

        let allocator = ManuallyDrop::new(allocator);

        Ok(Self {
            width,
            height,
            window,
            _entry: entry,
            instance,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            #[cfg(debug_assertions)]
            debug_callback,
            surface_loader,
            surface,
            physical_device,
            graphics_queue_index,
            present_queue_index,
            device,
            graphics_queue,
            present_queue,
            graphics_command_pool,
            swapchain_loader,
            swapchain,
            format,
            extent,
            swapchain_images,
            allocator,
            scene_render_pass,
            color_images,
            color_image_allocations,
            color_image_views,
            depth_images,
            depth_image_allocations,
            depth_image_views,
            scene_framebuffers,
            uniform_buffers,
            uniform_buffer_allocations,
            descriptor_pool,
            descriptor_set_layouts,
            descriptor_sets,
            graphics_pipeline,
            pipeline_layout,
            vertices,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),
            fences,
            current_frame: 0,
            image_available_semaphores,
            render_finished_semaphores,
            command_buffers,

            egui_integration,
            show_user_texture_window: false,
            show_scene_window: false,
            prev_pointer_pos: None,
            rotation_x: 0.0,
            rotation_y: 0.0,
            clear_color: [0.5, 0.25, 0.25, 1.0],
            light_position: Vector3::new(0.0, -16.0, -16.0),

            image,
            image_allocation: Some(image_allocation),
            image_view,
            sampler,
            image_texture_id,
            scene_texture_ids,
        })
    }

    fn draw(&mut self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }

        unsafe {
            // Wait Fence
            let fence = self.fences[self.current_frame];
            self.device.wait_for_fences(&[fence], true, std::u64::MAX)?;

            // Acquire next image
            let image_index;
            match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok((index, _is_suboptimal)) => image_index = index as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(error) => panic!("Error while acquiring next image: {}", error),
            }

            // Reset fence
            self.device.reset_fences(&[fence])?;

            // Setup command
            let command_buffer = self.command_buffers[image_index];

            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())?;

            // Clear framebuffer
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(self.swapchain_images[image_index])
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    )
                    .build()],
            );
            self.device.cmd_clear_color_image(
                command_buffer,
                self.swapchain_images[image_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
                &[vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1)
                    .build()],
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(self.swapchain_images[image_index])
                    .src_queue_family_index(self.graphics_queue_index)
                    .dst_queue_family_index(self.graphics_queue_index)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    )
                    .build()],
            );

            // Draw Scene Texture

            // Update UniformBuffer
            let uniform_buffer_object = UniformBufferObject {
                light: self.light_position.into(),
                model: (Matrix4::from_axis_angle(
                    Vector3::new(1.0, 0.0, 0.0),
                    Deg(self.rotation_x),
                ) * Matrix4::from_axis_angle(
                    Vector3::new(0.0, 1.0, 0.0),
                    Deg(self.rotation_y),
                ))
                .into(),
                view: Matrix4::look_at_rh(
                    Point3::new(0.0, -2.0, -5.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, -1.0, 0.0),
                )
                .into(),
                proj: Matrix4::perspective(
                    Deg(60.0),
                    SCENE_IMAGE_WIDTH as f32 / SCENE_IMAGE_HEIGHT as f32,
                    0.01,
                    100.0,
                )
                .into(),
            };
            let uniform_buffer_object_std140 = uniform_buffer_object.as_std140();
            {
                let ptr = self.uniform_buffer_allocations[image_index]
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut u8;
                ptr.copy_from_nonoverlapping(
                    uniform_buffer_object_std140.as_bytes().as_ptr(),
                    uniform_buffer_object_std140.as_bytes().len(),
                );
            }
            // Begin render pass
            self.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.scene_render_pass)
                    .framebuffer(self.scene_framebuffers[self.current_frame])
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(
                                vk::Extent2D::builder()
                                    .width(SCENE_IMAGE_WIDTH)
                                    .height(SCENE_IMAGE_HEIGHT)
                                    .build(),
                            )
                            .build(),
                    )
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: self.clear_color,
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            // Set Viewport
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::builder()
                    .width(SCENE_IMAGE_WIDTH as f32)
                    .height(SCENE_IMAGE_HEIGHT as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
                    .build()],
            );
            // Set Scissor
            self.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().build())
                    .extent(vk::Extent2D {
                        width: SCENE_IMAGE_WIDTH,
                        height: SCENE_IMAGE_HEIGHT,
                    })
                    .build()],
            );
            // Set Vertex buffer
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            // Set Descriptor Set
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
            // Draw Command
            self.device
                .cmd_draw(command_buffer, self.vertices.len() as u32, 1, 0, 0);
            // End render pass
            self.device.cmd_end_render_pass(command_buffer);
            // Pipeline barrier
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.color_images[self.current_frame])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build()],
            );

            // #### egui ##########################################################################
            self.egui_integration.begin_frame();
            egui::SidePanel::left("my_side_panel").show(&self.egui_integration.context(), |ui| {
                ui.heading("User Texture Example");
                ui.separator();
                ui.checkbox(&mut self.show_user_texture_window, "User Texture Window");
                ui.checkbox(&mut self.show_scene_window, "Scene Window");
                ui.horizontal(|ui| {
                    ui.label("Scene Clear Color");
                    let mut hsva = egui::color::Hsva::from_rgba_premultiplied(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3]);
                    egui::color_picker::color_edit_button_hsva(
                        ui,
                        &mut hsva,
                        egui::color_picker::Alpha::OnlyBlend,
                    );
                    self.clear_color = hsva.to_rgba_premultiplied();
                });
            });
            {
                let show_user_texture_window = &mut self.show_user_texture_window;
                let image_texture_id = self.image_texture_id;
                egui::Window::new("User Texture Window")
                    .resizable(true)
                    .scroll2([true, true])
                    .open(show_user_texture_window)
                    .show(&self.egui_integration.context(), |ui| {
                        ui.image(image_texture_id, [256.0, 256.0]);
                    });
            }
            {
                let show_scene_window = &mut self.show_scene_window;
                let scene_texture_ids = &self.scene_texture_ids;
                let current_frame = self.current_frame;
                let prev_pointer_pos = &mut self.prev_pointer_pos;
                let rotation_x = &mut self.rotation_x;
                let rotation_y = &mut self.rotation_y;
                egui::Window::new("Scene Window")
                    .resizable(true)
                    .scroll2([true, true])
                    .open(show_scene_window)
                    .show(&self.egui_integration.context(), |ui| {
                        {
                            use egui::*;

                            let desired_size =
                                vec2(SCENE_IMAGE_WIDTH as f32, SCENE_IMAGE_HEIGHT as f32);
                            let (rect, response) =
                                ui.allocate_exact_size(desired_size, Sense::drag());

                            let mut mesh =
                                epaint::Mesh::with_texture(scene_texture_ids[current_frame]);
                            mesh.add_rect_with_uv(
                                rect,
                                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                                Color32::WHITE,
                            );
                            ui.painter().add(Shape::Mesh(mesh));

                            if response.dragged() {
                                if ui.input().pointer.button_down(PointerButton::Primary) {
                                    if let Some(pointer_pos) = ui.input().pointer.interact_pos() {
                                        if let Some(prev_pointer_pos) = prev_pointer_pos {
                                            let delta = pointer_pos - *prev_pointer_pos;
                                            *rotation_x += delta.y * 2.0;
                                            *rotation_y += -delta.x * 2.0;
                                        }
                                        *prev_pointer_pos = Some(pointer_pos);
                                    }
                                }
                            }
                            if response.drag_released() {
                                *prev_pointer_pos = None;
                            }
                        }
                        ui.label("You can drag the scene to rotate the model.");
                    });
            }
            let (_, shapes) = self.egui_integration.end_frame(&mut self.window);
            let clipped_meshes = self.egui_integration.context().tessellate(shapes);
            self.egui_integration
                .paint(command_buffer, image_index, clipped_meshes);
            // #### egui ##########################################################################

            self.device.end_command_buffer(command_buffer)?;

            // Submit command
            self.device.queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::builder()
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[command_buffer])
                    .wait_semaphores(&[self.image_available_semaphores[self.current_frame]])
                    .signal_semaphores(&[self.render_finished_semaphores[self.current_frame]])
                    .build()],
                fence,
            )?;

            // Present Image
            match self.swapchain_loader.queue_present(
                self.present_queue,
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[self.swapchain])
                    .image_indices(&[image_index as u32])
                    .wait_semaphores(&[self.render_finished_semaphores[self.current_frame]]),
            ) {
                Ok(is_suboptimal) if is_suboptimal == true => return self.recreate_swapchain(),
                Ok(_is_suboptimal) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(error) => panic!("Failed to present queue: {}", error),
            }
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        self.width = size.width;
        self.height = size.height;

        // skip recreation if size is 0
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }

        // Wait idle
        unsafe {
            self.device.device_wait_idle()?;
        }

        // Recreate swapchain
        {
            let old_swapchain = self.swapchain;

            let capabilities = unsafe {
                self.surface_loader
                    .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
            };
            let formats = unsafe {
                self.surface_loader
                    .get_physical_device_surface_formats(self.physical_device, self.surface)?
            };
            let present_modes = unsafe {
                self.surface_loader
                    .get_physical_device_surface_present_modes(self.physical_device, self.surface)?
            };

            self.format = formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM
                })
                .unwrap_or(&formats[0])
                .clone();
            let present_mode = present_modes
                .into_iter()
                .find(|&p| p == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            self.extent = {
                if capabilities.current_extent.width != u32::max_value() {
                    capabilities.current_extent
                } else {
                    vk::Extent2D {
                        width: self
                            .width
                            .max(capabilities.min_image_extent.width)
                            .min(capabilities.max_image_extent.width),
                        height: self
                            .height
                            .max(capabilities.min_image_extent.height)
                            .min(capabilities.max_image_extent.height),
                    }
                }
            };

            let image_count = capabilities.min_image_count + 1;
            let image_count = if capabilities.max_image_count != 0 {
                image_count.min(capabilities.max_image_count)
            } else {
                image_count
            };

            let (image_sharing_mode, queue_family_indices) =
                if self.graphics_queue_index != self.present_queue_index {
                    (
                        vk::SharingMode::EXCLUSIVE,
                        vec![self.graphics_queue_index, self.present_queue_index],
                    )
                } else {
                    (vk::SharingMode::EXCLUSIVE, vec![])
                };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(self.format.format)
                .image_color_space(self.format.color_space)
                .image_extent(self.extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(queue_family_indices.as_slice())
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain);

            self.swapchain = unsafe {
                self.swapchain_loader
                    .create_swapchain(&swapchain_create_info, None)?
            };

            // destroy old swapchain
            unsafe {
                self.swapchain_loader.destroy_swapchain(old_swapchain, None);
            }
        };
        self.swapchain_images =
            unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain)? };

        // #### egui ##########################################################################
        self.egui_integration.update_swapchain(
            self.width,
            self.height,
            self.swapchain.clone(),
            self.format.clone(),
        );
        // #### egui ##########################################################################

        Ok(())
    }
}
impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.queue_wait_idle(self.graphics_queue).unwrap();
            self.device.queue_wait_idle(self.present_queue).unwrap();

            self.egui_integration
                .unregister_user_texture(self.image_texture_id);
            for &texture_id in self.scene_texture_ids.iter() {
                self.egui_integration.unregister_user_texture(texture_id);
            }

            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            if let Some(allocation) = self.image_allocation.take() {
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
            self.device.destroy_sampler(self.sampler, None);

            self.egui_integration.destroy();
            ManuallyDrop::drop(&mut self.egui_integration);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.fences[i], None);
            }
            self.device.destroy_buffer(self.vertex_buffer, None);
            if let Some(allocation) = self.vertex_buffer_allocation.take() {
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for &color_image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(color_image_view, None);
            }
            for &depth_image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(depth_image_view, None);
            }
            for &framebuffer in self.scene_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for (image, allocation) in self
                .depth_images
                .drain(0..)
                .zip(self.depth_image_allocations.drain(0..))
            {
                self.device.destroy_image(image, None);
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
            for (image, allocation) in self
                .color_images
                .drain(0..)
                .zip(self.color_image_allocations.drain(0..))
            {
                self.device.destroy_image(image, None);
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
            self.device
                .destroy_render_pass(self.scene_render_pass, None);
            for (buffer, allocation) in self
                .uniform_buffers
                .drain(0..)
                .zip(self.uniform_buffer_allocations.drain(0..))
            {
                self.device.destroy_buffer(buffer, None);
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
            for &layout in self.descriptor_set_layouts.iter() {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);

            ManuallyDrop::drop(&mut self.allocator);

            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            #[cfg(debug_assertions)]
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    let mut app = App::new(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        app.egui_integration.handle_event(&event);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => app.recreate_swapchain().unwrap(),
            Event::MainEventsCleared => app.window.request_redraw(),
            Event::RedrawRequested(_window_id) => app.draw().unwrap(),
            _ => (),
        }
    })
}
