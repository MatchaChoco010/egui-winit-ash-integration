#![warn(missing_docs)]

use std::ffi::CString;
use std::include_bytes;
use std::time::Instant;

use ash::{extensions::khr::Swapchain, vk, Device};
use bytemuck::bytes_of;
use copypasta::{ClipboardContext, ClipboardProvider};
use egui::{
    emath::{pos2, vec2},
    epaint::ClippedShape,
    CtxRef, Key,
};
use winit::event::{Event, ModifiersState, VirtualKeyCode, WindowEvent};
use winit::window::Window;

use crate::*;

/// egui integration with winit and ash.
pub struct Integration<A: AllocatorTrait> {
    start_time: Option<Instant>,

    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    context: CtxRef,
    raw_input: egui::RawInput,
    mouse_pos: egui::Pos2,
    modifiers_state: ModifiersState,
    clipboard: ClipboardContext,
    current_cursor_icon: egui::CursorIcon,

    device: Device,
    allocator: A,
    swapchain_loader: Swapchain,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    framebuffer_color_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<A::Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<A::Allocation>,
    font_image_staging_buffer: vk::Buffer,
    font_image_staging_buffer_allocation: Option<A::Allocation>,
    font_image: vk::Image,
    font_image_allocation: Option<A::Allocation>,
    font_image_view: vk::ImageView,
    font_image_size: (u64, u64),
    font_image_version: u64,
    font_descriptor_sets: Vec<vk::DescriptorSet>,

    user_texture_layout: vk::DescriptorSetLayout,
    user_textures: Vec<Option<vk::DescriptorSet>>,
}
impl<A: AllocatorTrait> Integration<A> {
    /// Create an instance of the integration.
    pub fn new(
        physical_width: u32,
        physical_height: u32,
        scale_factor: f64,
        font_definitions: egui::FontDefinitions,
        style: egui::Style,
        device: Device,
        allocator: A,
        swapchain_loader: Swapchain,
        swapchain: vk::SwapchainKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) -> Self {
        // Start time is initialized when first time call render_time
        let start_time = None;

        // Create context
        let context = CtxRef::default();
        context.set_fonts(font_definitions);
        context.set_style(style);

        // Create raw_input
        let raw_input = egui::RawInput {
            pixels_per_point: Some(scale_factor as f32),
            screen_rect: Some(egui::Rect::from_min_size(
                Default::default(),
                vec2(physical_width as f32, physical_height as f32) / scale_factor as f32,
            )),
            time: Some(0.0),
            ..Default::default()
        };

        // Create mouse pos and modifier state (These values are overwritten by handle events)
        let mouse_pos = pos2(0.0, 0.0);
        let modifiers_state = winit::event::ModifiersState::default();

        // Create clipboard context
        let clipboard = ClipboardContext::new().expect("Failed to initialize ClipboardContext.");

        // Get swap_images to get len of swapchain images and to create framebuffers
        let swap_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images.")
        };

        // Create DescriptorPool
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(1024)
                    .pool_sizes(&[vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1024)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create descriptor pool.");

        // Create DescriptorSetLayouts
        let descriptor_set_layouts = {
            let mut sets = vec![];
            for _ in 0..swap_images.len() {
                sets.push(
                    unsafe {
                        device.create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                                vk::DescriptorSetLayoutBinding::builder()
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .descriptor_count(1)
                                    .binding(0)
                                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                                    .build(),
                            ]),
                            None,
                        )
                    }
                    .expect("Failed to create descriptor set layout."),
                );
            }
            sets
        };

        // Create RenderPass
        let render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[vk::AttachmentDescription::builder()
                        .format(surface_format.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .build()])
                    .subpasses(&[vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::builder()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .build()])
                        .build()])
                    .dependencies(&[vk::SubpassDependency::builder()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create render pass.");

        // Create PipelineLayout
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&[vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(std::mem::size_of::<f32>() as u32 * 2) // screen size
                        .build()]),
                None,
            )
        }
        .expect("Failed to create pipeline layout.");

        // Create Pipeline
        let pipeline = {
            let bindings = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(
                    4 * std::mem::size_of::<f32>() as u32 + 4 * std::mem::size_of::<u8>() as u32,
                )
                .build()];

            let attributes = [
                // position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // uv
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // color
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .build(),
            ];

            let vertex_shader_module = {
                let bytes_code = include_bytes!("shaders/spv/vert.spv");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe { device.create_shader_module(&shader_module_create_info, None) }
                    .expect("Failed to create vertex shader module.")
            };
            let fragment_shader_module = {
                let bytes_code = include_bytes!("shaders/spv/frag.spv");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe { device.create_shader_module(&shader_module_create_info, None) }
                    .expect("Failed to create fragment shader module.")
            };
            let main_function_name = CString::new("main").unwrap();
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

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::ALWAYS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .build()];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

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
                .render_pass(render_pass)
                .subpass(0)
                .build()];

            let pipeline = unsafe {
                device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_info,
                    None,
                )
            }
            .expect("Failed to create graphics pipeline.")[0];
            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }
            pipeline
        };

        // Create Sampler
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .expect("Failed to create sampler.");

        // Create Framebuffers
        let framebuffer_color_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(swapchain_image.clone())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format.format)
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
                    )
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();
        let framebuffers = framebuffer_color_image_views
            .iter()
            .map(|&image_views| unsafe {
                let attachments = &[image_views];
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(render_pass)
                            .attachments(attachments)
                            .width(physical_width)
                            .height(physical_height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer.")
            })
            .collect::<Vec<_>>();

        // Create vertex buffer and index buffer
        let mut vertex_buffers = vec![];
        let mut vertex_buffer_allocations = vec![];
        let mut index_buffers = vec![];
        let mut index_buffer_allocations = vec![];
        for _ in 0..framebuffers.len() {
            let vertex_buffer = unsafe {
                device
                    .create_buffer(
                        &vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::vertex_buffer_size()),
                        None,
                    )
                    .expect("Failed to create vertex buffer.")
            };
            let vertex_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = allocator
                .allocate(A::AllocationCreateInfo::new(
                    vertex_buffer_requirements,
                    MemoryLocation::CpuToGpu,
                    true,
                ))
                .expect("Failed to create vertex buffer.");
            unsafe {
                device
                    .bind_buffer_memory(
                        vertex_buffer,
                        vertex_buffer_allocation.memory(),
                        vertex_buffer_allocation.offset(),
                    )
                    .expect("Failed to create vertex buffer.")
            }

            let index_buffer = unsafe {
                device
                    .create_buffer(
                        &vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::index_buffer_size()),
                        None,
                    )
                    .expect("Failed to create index buffer.")
            };
            let index_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(index_buffer) };
            let index_buffer_allocation = allocator
                .allocate(A::AllocationCreateInfo::new(
                    index_buffer_requirements,
                    MemoryLocation::CpuToGpu,
                    true,
                ))
                .expect("Failed to create index buffer.");
            unsafe {
                device
                    .bind_buffer_memory(
                        index_buffer,
                        index_buffer_allocation.memory(),
                        index_buffer_allocation.offset(),
                    )
                    .expect("Failed to create index buffer.")
            }

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_allocations.push(vertex_buffer_allocation);
            index_buffers.push(index_buffer);
            index_buffer_allocations.push(index_buffer_allocation);
        }

        // Create font image and anything related to it
        // These values will be uploaded at rendering time
        let font_image_staging_buffer = Default::default();
        let font_image_staging_buffer_allocation = None;
        let font_image = Default::default();
        let font_image_allocation = None;
        let font_image_view = Default::default();
        let font_image_size = (0, 0);
        let font_image_version = 0;
        let font_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&descriptor_set_layouts),
            )
        }
        .expect("Failed to create descriptor sets.");

        // User Textures
        let user_texture_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                    vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .binding(0)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .build(),
                ]),
                None,
            )
        }
        .expect("Failed to create descriptor set layout.");
        let user_textures = vec![];

        Self {
            start_time,

            physical_width,
            physical_height,
            scale_factor,
            context,
            raw_input,
            mouse_pos,
            modifiers_state,
            clipboard,
            current_cursor_icon: egui::CursorIcon::None,

            device,
            allocator,
            swapchain_loader,
            descriptor_pool,
            descriptor_set_layouts,
            pipeline_layout,
            pipeline,
            sampler,
            render_pass,
            framebuffer_color_image_views,
            framebuffers,
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
            font_image_staging_buffer,
            font_image_staging_buffer_allocation,
            font_image,
            font_image_allocation,
            font_image_view,
            font_image_size,
            font_image_version,
            font_descriptor_sets,

            user_texture_layout,
            user_textures,
        }
    }

    // vertex buffer size
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    // index buffer size
    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    /// handling winit event.
    pub fn handle_event<T>(&mut self, winit_event: &Event<T>) {
        match winit_event {
            Event::WindowEvent {
                window_id: _window_id,
                event,
            } => match event {
                // window size changed
                WindowEvent::Resized(physical_size) => {
                    let pixels_per_point = self
                        .raw_input
                        .pixels_per_point
                        .unwrap_or_else(|| self.context.pixels_per_point());
                    self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        Default::default(),
                        vec2(physical_size.width as f32, physical_size.height as f32)
                            / pixels_per_point,
                    ));
                }
                // dpi changed
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    self.scale_factor = *scale_factor;
                    self.raw_input.pixels_per_point = Some(*scale_factor as f32);
                    let pixels_per_point = self
                        .raw_input
                        .pixels_per_point
                        .unwrap_or_else(|| self.context.pixels_per_point());
                    self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        Default::default(),
                        vec2(new_inner_size.width as f32, new_inner_size.height as f32)
                            / pixels_per_point,
                    ));
                }
                // mouse click
                WindowEvent::MouseInput { state, button, .. } => {
                    if let Some(button) = Self::winit_to_egui_mouse_button(*button) {
                        self.raw_input.events.push(egui::Event::PointerButton {
                            pos: self.mouse_pos,
                            button,
                            pressed: *state == winit::event::ElementState::Pressed,
                            modifiers: Self::winit_to_egui_modifiers(self.modifiers_state),
                        });
                    }
                }
                // mouse wheel
                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => {
                        let line_height = 24.0;
                        self.raw_input.events.push(egui::Event::Scroll(vec2(*x, *y) * line_height));
                    }
                    winit::event::MouseScrollDelta::PixelDelta(delta) => {
                        self.raw_input.events.push(egui::Event::Scroll(vec2(delta.x as f32, delta.y as f32)));
                    }
                },
                // mouse move
                WindowEvent::CursorMoved { position, .. } => {
                    let pixels_per_point = self
                        .raw_input
                        .pixels_per_point
                        .unwrap_or_else(|| self.context.pixels_per_point());
                    let pos = pos2(
                        position.x as f32 / pixels_per_point,
                        position.y as f32 / pixels_per_point,
                    );
                    self.raw_input.events.push(egui::Event::PointerMoved(pos));
                    self.mouse_pos = pos;
                }
                // mouse out
                WindowEvent::CursorLeft { .. } => {
                    self.raw_input.events.push(egui::Event::PointerGone);
                }
                // modifier keys
                WindowEvent::ModifiersChanged(input) => self.modifiers_state = *input,
                // keyboard inputs
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(virtual_keycode) = input.virtual_keycode {
                        let pressed = input.state == winit::event::ElementState::Pressed;
                        if pressed {
                            let is_ctrl = self.modifiers_state.ctrl();
                            if is_ctrl && virtual_keycode == VirtualKeyCode::C {
                                self.raw_input.events.push(egui::Event::Copy);
                            } else if is_ctrl && virtual_keycode == VirtualKeyCode::X {
                                self.raw_input.events.push(egui::Event::Cut);
                            } else if is_ctrl && virtual_keycode == VirtualKeyCode::V {
                                if let Ok(contents) = self.clipboard.get_contents() {
                                    self.raw_input.events.push(egui::Event::Text(contents));
                                }
                            } else if let Some(key) = Self::winit_to_egui_key_code(virtual_keycode)
                            {
                                self.raw_input.events.push(egui::Event::Key {
                                    key,
                                    pressed: input.state == winit::event::ElementState::Pressed,
                                    modifiers: Self::winit_to_egui_modifiers(self.modifiers_state),
                                })
                            }
                        }
                    }
                }
                // receive character
                WindowEvent::ReceivedCharacter(ch) => {
                    // remove control character
                    if ch.is_ascii_control() {
                        return;
                    }
                    self.raw_input
                        .events
                        .push(egui::Event::Text(ch.to_string()));
                }
                _ => (),
            },
            _ => (),
        }
    }

    fn winit_to_egui_key_code(key: VirtualKeyCode) -> Option<egui::Key> {
        Some(match key {
            VirtualKeyCode::Down => Key::ArrowDown,
            VirtualKeyCode::Left => Key::ArrowLeft,
            VirtualKeyCode::Right => Key::ArrowRight,
            VirtualKeyCode::Up => Key::ArrowUp,
            VirtualKeyCode::Escape => Key::Escape,
            VirtualKeyCode::Tab => Key::Tab,
            VirtualKeyCode::Back => Key::Backspace,
            VirtualKeyCode::Return => Key::Enter,
            VirtualKeyCode::Space => Key::Space,
            VirtualKeyCode::Insert => Key::Insert,
            VirtualKeyCode::Delete => Key::Delete,
            VirtualKeyCode::Home => Key::Home,
            VirtualKeyCode::End => Key::End,
            VirtualKeyCode::PageUp => Key::PageUp,
            VirtualKeyCode::PageDown => Key::PageDown,
            VirtualKeyCode::Key0 => Key::Num0,
            VirtualKeyCode::Key1 => Key::Num1,
            VirtualKeyCode::Key2 => Key::Num2,
            VirtualKeyCode::Key3 => Key::Num3,
            VirtualKeyCode::Key4 => Key::Num4,
            VirtualKeyCode::Key5 => Key::Num5,
            VirtualKeyCode::Key6 => Key::Num6,
            VirtualKeyCode::Key7 => Key::Num7,
            VirtualKeyCode::Key8 => Key::Num8,
            VirtualKeyCode::Key9 => Key::Num9,
            VirtualKeyCode::A => Key::A,
            VirtualKeyCode::B => Key::B,
            VirtualKeyCode::C => Key::C,
            VirtualKeyCode::D => Key::D,
            VirtualKeyCode::E => Key::E,
            VirtualKeyCode::F => Key::F,
            VirtualKeyCode::G => Key::G,
            VirtualKeyCode::H => Key::H,
            VirtualKeyCode::I => Key::I,
            VirtualKeyCode::J => Key::J,
            VirtualKeyCode::K => Key::K,
            VirtualKeyCode::L => Key::L,
            VirtualKeyCode::M => Key::M,
            VirtualKeyCode::N => Key::N,
            VirtualKeyCode::O => Key::O,
            VirtualKeyCode::P => Key::P,
            VirtualKeyCode::Q => Key::Q,
            VirtualKeyCode::R => Key::R,
            VirtualKeyCode::S => Key::S,
            VirtualKeyCode::T => Key::T,
            VirtualKeyCode::U => Key::U,
            VirtualKeyCode::V => Key::V,
            VirtualKeyCode::W => Key::W,
            VirtualKeyCode::X => Key::X,
            VirtualKeyCode::Y => Key::Y,
            VirtualKeyCode::Z => Key::Z,
            _ => return None,
        })
    }

    fn winit_to_egui_modifiers(modifiers: ModifiersState) -> egui::Modifiers {
        egui::Modifiers {
            alt: modifiers.alt(),
            ctrl: modifiers.ctrl(),
            shift: modifiers.shift(),
            #[cfg(target_os = "macos")]
            mac_cmd: modifiers.logo(),
            #[cfg(target_os = "macos")]
            command: modifiers.logo(),
            #[cfg(not(target_os = "macos"))]
            mac_cmd: false,
            #[cfg(not(target_os = "macos"))]
            command: modifiers.ctrl(),
        }
    }

    fn winit_to_egui_mouse_button(
        button: winit::event::MouseButton,
    ) -> Option<egui::PointerButton> {
        Some(match button {
            winit::event::MouseButton::Left => egui::PointerButton::Primary,
            winit::event::MouseButton::Right => egui::PointerButton::Secondary,
            winit::event::MouseButton::Middle => egui::PointerButton::Middle,
            _ => return None,
        })
    }

    /// Convert from [`egui::CursorIcon`] to [`winit::window::CursorIcon`].
    fn egui_to_winit_cursor_icon(
        cursor_icon: egui::CursorIcon,
    ) -> Option<winit::window::CursorIcon> {
        Some(match cursor_icon {
            egui::CursorIcon::Default => winit::window::CursorIcon::Default,
            egui::CursorIcon::PointingHand => winit::window::CursorIcon::Hand,
            egui::CursorIcon::ResizeHorizontal => winit::window::CursorIcon::ColResize,
            egui::CursorIcon::ResizeNeSw => winit::window::CursorIcon::NeResize,
            egui::CursorIcon::ResizeNwSe => winit::window::CursorIcon::NwResize,
            egui::CursorIcon::ResizeVertical => winit::window::CursorIcon::RowResize,
            egui::CursorIcon::Text => winit::window::CursorIcon::Text,
            egui::CursorIcon::Grab => winit::window::CursorIcon::Grab,
            egui::CursorIcon::Grabbing => winit::window::CursorIcon::Grabbing,
            egui::CursorIcon::None => return None,
            egui::CursorIcon::ContextMenu => winit::window::CursorIcon::ContextMenu,
            egui::CursorIcon::Help => winit::window::CursorIcon::Help,
            egui::CursorIcon::Progress => winit::window::CursorIcon::Progress,
            egui::CursorIcon::Wait => winit::window::CursorIcon::Wait,
            egui::CursorIcon::Cell => winit::window::CursorIcon::Cell,
            egui::CursorIcon::Crosshair => winit::window::CursorIcon::Crosshair,
            egui::CursorIcon::VerticalText => winit::window::CursorIcon::VerticalText,
            egui::CursorIcon::Alias => winit::window::CursorIcon::Alias,
            egui::CursorIcon::Copy => winit::window::CursorIcon::Copy,
            egui::CursorIcon::Move => winit::window::CursorIcon::Move,
            egui::CursorIcon::NoDrop => winit::window::CursorIcon::NoDrop,
            egui::CursorIcon::NotAllowed => winit::window::CursorIcon::NotAllowed,
            egui::CursorIcon::AllScroll => winit::window::CursorIcon::AllScroll,
            egui::CursorIcon::ZoomIn => winit::window::CursorIcon::ZoomIn,
            egui::CursorIcon::ZoomOut => winit::window::CursorIcon::ZoomOut,
        })
    }

    /// begin frame.
    pub fn begin_frame(&mut self) {
        self.context.begin_frame(self.raw_input.take());
    }

    /// end frame.
    pub fn end_frame(&mut self, window: &Window) -> (egui::Output, Vec<ClippedShape>) {
        let (output, clipped_shapes) = self.context.end_frame();

        // handle links
        if let Some(egui::output::OpenUrl { url, .. }) = &output.open_url {
            if let Err(err) = webbrowser::open(url) {
                eprintln!("Failed to open url: {}", err);
            }
        }

        // handle clipboard
        if !output.copied_text.is_empty() {
            if let Err(err) = self.clipboard.set_contents(output.copied_text.clone()) {
                eprintln!("Copy/Cut error: {}", err);
            }
        }

        // handle cursor icon
        if self.current_cursor_icon != output.cursor_icon {
            if let Some(cursor_icon) =
                Integration::<A>::egui_to_winit_cursor_icon(output.cursor_icon)
            {
                window.set_cursor_visible(true);
                window.set_cursor_icon(cursor_icon);
            } else {
                window.set_cursor_visible(false);
            }
            self.current_cursor_icon = output.cursor_icon;
        }

        (output, clipped_shapes)
    }

    /// Get [`egui::CtxRef`].
    pub fn context(&self) -> CtxRef {
        self.context.clone()
    }

    /// Record paint commands.
    pub fn paint(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain_image_index: usize,
        clipped_meshes: Vec<egui::ClippedMesh>,
    ) {
        let index = swapchain_image_index;

        // update time
        if let Some(time) = self.start_time {
            self.raw_input.time = Some(time.elapsed().as_secs_f64());
        } else {
            self.start_time = Some(Instant::now());
        }

        // update font texture
        self.upload_font_texture(command_buffer, &self.context.fonts().font_image());

        let mut vertex_buffer_ptr = self.vertex_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8;
        // let mut vertex_buffer_ptr = unsafe {
        //     self.device
        //         .map_memory(
        //             self.vertex_buffer_allocations[index].memory(),
        //             self.vertex_buffer_allocations[index].offset(),
        //             self.vertex_buffer_allocations[index].size(),
        //             vk::MemoryMapFlags::empty(),
        //         )
        //         .expect("Failed to map buffers.") as *mut u8
        // };
        let vertex_buffer_ptr_end =
            unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
        // let mut index_buffer_ptr = unsafe {
        //     self.device
        //         .map_memory(
        //             self.index_buffer_allocations[index].memory(),
        //             self.index_buffer_allocations[index].offset(),
        //             self.index_buffer_allocations[index].size(),
        //             vk::MemoryMapFlags::empty(),
        //         )
        //         .expect("Failed to map buffers.") as *mut u8
        // };
        let mut index_buffer_ptr = self.index_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8;
        let index_buffer_ptr_end =
            unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };

        // begin render pass
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .clear_values(&[])
                    .render_area(
                        vk::Rect2D::builder()
                            .extent(
                                vk::Extent2D::builder()
                                    .width(self.physical_width)
                                    .height(self.physical_height)
                                    .build(),
                            )
                            .build(),
                    ),
                vk::SubpassContents::INLINE,
            );
        }

        // bind resources
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffers[index]],
                &[0],
            );
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffers[index],
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(self.physical_width as f32)
                    .height(self.physical_height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
                    .build()],
            );
            let width_points = self.physical_width as f32 / self.scale_factor as f32;
            let height_points = self.physical_height as f32 / self.scale_factor as f32;
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes_of(&width_points),
            );
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                std::mem::size_of_val(&width_points) as u32,
                bytes_of(&height_points),
            );
        }

        // render meshes
        let mut vertex_base = 0;
        let mut index_base = 0;
        for egui::ClippedMesh(rect, mesh) in clipped_meshes {
            // update texture
            unsafe {
                if let egui::TextureId::User(id) = mesh.texture_id {
                    if let Some(descriptor_set) = self.user_textures[id as usize] {
                        self.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                    } else {
                        eprintln!(
                            "This UserTexture has already been unregistered: {:?}",
                            mesh.texture_id
                        );
                        continue;
                    }
                } else {
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.font_descriptor_sets[index]],
                        &[],
                    );
                }
            }

            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            let v_slice = &mesh.vertices;
            let v_size = std::mem::size_of_val(&v_slice[0]);
            let v_copy_size = v_slice.len() * v_size;

            let i_slice = &mesh.indices;
            let i_size = std::mem::size_of_val(&i_slice[0]);
            let i_copy_size = i_slice.len() * i_size;

            let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_size) };
            let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_size) };

            if vertex_buffer_ptr_next >= vertex_buffer_ptr_end
                || index_buffer_ptr_next >= index_buffer_ptr_end
            {
                panic!("egui paint out of memory");
            }

            // map memory
            unsafe { vertex_buffer_ptr.copy_from(v_slice.as_ptr() as *const u8, v_copy_size) };
            unsafe { index_buffer_ptr.copy_from(i_slice.as_ptr() as *const u8, i_copy_size) };

            vertex_buffer_ptr = vertex_buffer_ptr_next;
            index_buffer_ptr = index_buffer_ptr_next;

            // record draw commands
            unsafe {
                let min = rect.min;
                let min = egui::Pos2 {
                    x: min.x * self.scale_factor as f32,
                    y: min.y * self.scale_factor as f32,
                };
                let min = egui::Pos2 {
                    x: f32::clamp(min.x, 0.0, self.physical_width as f32),
                    y: f32::clamp(min.y, 0.0, self.physical_height as f32),
                };
                let max = rect.max;
                let max = egui::Pos2 {
                    x: max.x * self.scale_factor as f32,
                    y: max.y * self.scale_factor as f32,
                };
                let max = egui::Pos2 {
                    x: f32::clamp(max.x, min.x, self.physical_width as f32),
                    y: f32::clamp(max.y, min.y, self.physical_height as f32),
                };
                self.device.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D::builder()
                        .offset(
                            vk::Offset2D::builder()
                                .x(min.x.round() as i32)
                                .y(min.y.round() as i32)
                                .build(),
                        )
                        .extent(
                            vk::Extent2D::builder()
                                .width((max.x.round() - min.x) as u32)
                                .height((max.y.round() - min.y) as u32)
                                .build(),
                        )
                        .build()],
                );
                self.device.cmd_draw_indexed(
                    command_buffer,
                    mesh.indices.len() as u32,
                    1,
                    index_base,
                    vertex_base,
                    0,
                );
            }

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }

        // end render pass
        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
        }
    }

    fn upload_font_texture(&mut self, command_buffer: vk::CommandBuffer, texture: &egui::FontImage) {
        debug_assert_eq!(texture.pixels.len(), texture.width * texture.height);

        // check version
        if texture.version == self.font_image_version {
            return;
        }

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        let dimensions = (texture.width as u64, texture.height as u64);
        let data = texture
            .pixels
            .iter()
            .flat_map(|&r| vec![r, r, r, r])
            .collect::<Vec<_>>();

        // free prev staging buffer
        if let Some(allocation) = self.font_image_staging_buffer_allocation.take() {
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        unsafe {
            self.device
                .destroy_buffer(self.font_image_staging_buffer, None);
        }

        // free font image
        unsafe {
            self.device.destroy_image_view(self.font_image_view, None);
        }
        if let Some(allocation) = self.font_image_allocation.take() {
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        unsafe {
            self.device.destroy_image(self.font_image, None);
        }

        // create font image
        let font_image_staging_buffer = unsafe {
            self.device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .size(dimensions.0 * dimensions.1 * 4),
                    None,
                )
                .expect("Failed to create buffer.")
        };
        let font_image_staging_buffer_requirements = unsafe {
            self.device
                .get_buffer_memory_requirements(font_image_staging_buffer)
        };
        let font_image_staging_buffer_allocation = self
            .allocator
            .allocate(A::AllocationCreateInfo::new(
                font_image_staging_buffer_requirements,
                MemoryLocation::CpuToGpu,
                true,
            ))
            .expect("Failed to create buffer.");
        unsafe {
            self.device
                .bind_buffer_memory(
                    font_image_staging_buffer,
                    font_image_staging_buffer_allocation.memory(),
                    font_image_staging_buffer_allocation.offset(),
                )
                .expect("Failed to create buffer.")
        }
        self.font_image_staging_buffer = font_image_staging_buffer;
        self.font_image_staging_buffer_allocation = Some(font_image_staging_buffer_allocation);
        let font_image = unsafe {
            self.device
                .create_image(
                    &vk::ImageCreateInfo::builder()
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .image_type(vk::ImageType::TYPE_2D)
                        .mip_levels(1)
                        .array_layers(1)
                        .extent(vk::Extent3D {
                            width: dimensions.0 as u32,
                            height: dimensions.1 as u32,
                            depth: 1,
                        }),
                    None,
                )
                .expect("Failed to create image.")
        };
        let font_image_requirements =
            unsafe { self.device.get_image_memory_requirements(font_image) };
        let font_image_allocation = self
            .allocator
            .allocate(A::AllocationCreateInfo::new(
                font_image_requirements,
                MemoryLocation::GpuOnly,
                false,
            ))
            .expect("Failed to create image.");
        unsafe {
            self.device
                .bind_image_memory(
                    font_image,
                    font_image_allocation.memory(),
                    font_image_allocation.offset(),
                )
                .expect("Failed to create image.")
        }
        self.font_image = font_image;
        self.font_image_allocation = Some(font_image_allocation);
        self.font_image_view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(self.font_image)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .base_mip_level(0)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    ),
                None,
            )
        }
        .expect("Failed to create image view.");
        self.font_image_size = dimensions;
        self.font_image_version = texture.version;

        // update descriptor set
        for &font_descriptor_set in self.font_descriptor_sets.iter() {
            unsafe {
                self.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(font_descriptor_set)
                        .image_info(&[vk::DescriptorImageInfo::builder()
                            .image_view(self.font_image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .sampler(self.sampler)
                            .build()])
                        .dst_binding(0)
                        .build()],
                    &[],
                );
            }
        }

        // map memory
        if let Some(allocation) = &self.font_image_staging_buffer_allocation {
            let ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            unsafe {
                ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
            }
        }

        // record buffer staging commands to command buffer
        unsafe {
            // update image layout to transfer dst optimal
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .image(self.font_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .build(),
                    )
                    .src_access_mask(vk::AccessFlags::default())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .build()],
            );

            // copy staging buffer to image
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                self.font_image_staging_buffer,
                self.font_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::builder()
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                            .build(),
                    )
                    .image_extent(
                        vk::Extent3D::builder()
                            .width(dimensions.0 as u32)
                            .height(dimensions.1 as u32)
                            .depth(1)
                            .build(),
                    )
                    .build()],
            );

            // update image layout to shader read only optimal
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .image(self.font_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .build(),
                    )
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build()],
            );
        }
    }

    /// Update swapchain.
    pub fn update_swapchain(
        &mut self,
        physical_width: u32,
        physical_height: u32,
        swapchain: vk::SwapchainKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) {
        self.physical_width = physical_width;
        self.physical_height = physical_height;

        // release vk objects to be regenerated.
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_pipeline(self.pipeline, None);
            for &image_view in self.framebuffer_color_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
        }

        // swap images
        let swap_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images.");

        // Recreate render pass for update surface format
        self.render_pass = unsafe {
            self.device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[vk::AttachmentDescription::builder()
                        .format(surface_format.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .build()])
                    .subpasses(&[vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::builder()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .build()])
                        .build()])
                    .dependencies(&[vk::SubpassDependency::builder()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create render pass.");

        // Recreate pipeline for update render pass
        self.pipeline = {
            let bindings = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(5 * std::mem::size_of::<f32>() as u32)
                .build()];
            let attributes = [
                // position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // uv
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // color
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .build(),
            ];

            let vertex_shader_module = {
                let bytes_code = include_bytes!("shaders/spv/vert.spv");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create vertex shader module.")
            };
            let fragment_shader_module = {
                let bytes_code = include_bytes!("shaders/spv/frag.spv");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create fragment shader module.")
            };
            let main_function_name = CString::new("main").unwrap();
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

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .build()];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

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
                .layout(self.pipeline_layout)
                .render_pass(self.render_pass)
                .subpass(0)
                .build()];

            let pipeline = unsafe {
                self.device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_info,
                    None,
                )
            }
            .expect("Failed to create graphics pipeline")[0];
            unsafe {
                self.device
                    .destroy_shader_module(vertex_shader_module, None);
                self.device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            pipeline
        };

        // Recreate color image views for new framebuffers
        self.framebuffer_color_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                self.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(swapchain_image.clone())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format.format)
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
                    )
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();
        // Recreate framebuffers for new swapchain
        self.framebuffers = self
            .framebuffer_color_image_views
            .iter()
            .map(|&image_views| unsafe {
                let attachments = &[image_views];
                self.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(self.render_pass)
                            .attachments(attachments)
                            .width(physical_width)
                            .height(physical_height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer.")
            })
            .collect::<Vec<_>>();
    }

    /// Registering user texture.
    ///
    /// Pass the Vulkan ImageView and Sampler.
    /// `image_view`'s image layout must be `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// UserTexture needs to be unregistered when it is no longer needed.
    ///
    /// # Example
    /// ```sh
    /// cargo run --example user_texture
    /// ```
    /// [The example for user texture is in examples directory](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem/tree/main/examples/user_texture)
    pub fn register_user_texture(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> egui::TextureId {
        // get texture id
        let mut id = None;
        for (i, user_texture) in self.user_textures.iter().enumerate() {
            if user_texture.is_none() {
                id = Some(i as u64);
                break;
            }
        }
        let id = if let Some(i) = id {
            i
        } else {
            self.user_textures.len() as u64
        };

        // allocate and update descriptor set
        let layouts = [self.user_texture_layout];
        let descriptor_set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("Failed to create descriptor sets.")[0];
        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)
                        .build()])
                    .dst_binding(0)
                    .build()],
                &[],
            );
        }

        if id == self.user_textures.len() as u64 {
            self.user_textures.push(Some(descriptor_set));
        } else {
            self.user_textures[id as usize] = Some(descriptor_set);
        }

        egui::TextureId::User(id)
    }

    /// Unregister user texture.
    ///
    /// The internal texture (egui::TextureId::Egui) cannot be unregistered.
    pub fn unregister_user_texture(&mut self, texture_id: egui::TextureId) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(descriptor_set) = self.user_textures[id as usize] {
                unsafe {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .expect("Failed to free descriptor sets.");
                }
                self.user_textures[id as usize] = None;
            }
        } else {
            eprintln!("The internal texture cannot be unregistered; please pass the texture ID of UserTexture.");
            return;
        }
    }

    /// destroy vk objects.
    ///
    /// # Unsafe
    /// This method release vk objects memory that is not managed by Rust.
    pub unsafe fn destroy(&mut self) {
        self.device
            .destroy_descriptor_set_layout(self.user_texture_layout, None);
        self.device.destroy_image_view(self.font_image_view, None);
        self.device.destroy_image(self.font_image, None);
        if let Some(allocation) = self.font_image_allocation.take() {
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        self.device
            .destroy_buffer(self.font_image_staging_buffer, None);
        if let Some(allocation) = self.font_image_staging_buffer_allocation.take() {
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        for (buffer, allocation) in self
            .index_buffers
            .drain(0..)
            .zip(self.index_buffer_allocations.drain(0..))
        {
            self.device.destroy_buffer(buffer, None);
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        for (buffer, allocation) in self
            .vertex_buffers
            .drain(0..)
            .zip(self.vertex_buffer_allocations.drain(0..))
        {
            self.device.destroy_buffer(buffer, None);
            self.allocator
                .free(allocation)
                .expect("Failed to free allocation");
        }
        for &image_view in self.framebuffer_color_image_views.iter() {
            self.device.destroy_image_view(image_view, None);
        }
        for &framebuffer in self.framebuffers.iter() {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.device.destroy_render_pass(self.render_pass, None);
        self.device.destroy_sampler(self.sampler, None);
        self.device.destroy_pipeline(self.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.pipeline_layout, None);
        for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
            self.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);
    }
}
