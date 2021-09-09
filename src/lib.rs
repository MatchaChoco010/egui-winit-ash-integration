//! This is the [egui](https://github.com/emilk/egui) integration crate for
//! [winit](https://github.com/rust-windowing/winit) and [ash](https://github.com/MaikKlein/ash).
//! The default GPU allocator is
//! [gpu_allocator](https://github.com/Traverse-Research/gpu-allocator),
//! but you can also implement AllocatorTrait.
//!
//! # Example
//! ```sh
//! cargo run --example example
//! ```
//!
//! ```sh
//! cargo run --example user_texture
//! ```
//!
//! # Usage
//!
//! ```
//! fn main() -> Result<()> {
//!     let event_loop = EventLoop::new();
//!     // (1) Call Integration::<Arc<Mutex<Allocator>>>::new() in App::new().
//!     let mut app = App::new(&event_loop)?;
//!
//!     event_loop.run(move |event, _, control_flow| {
//!         *control_flow = ControlFlow::Poll;
//!         // (2) Call integration.handle_event(&event).
//!         app.egui_integration.handle_event(&event);
//!         match event {
//!             Event::WindowEvent {
//!                 event: WindowEvent::CloseRequested,
//!                 ..
//!             } => *control_flow = ControlFlow::Exit,
//!             Event::WindowEvent {
//!                 event: WindowEvent::Resized(_),
//!                 ..
//!             } => {
//!                 // (3) Call integration.recreate_swapchain(...) in app.recreate_swapchain().
//!                 app.recreate_swapchain().unwrap();
//!             }
//!             Event::MainEventsCleared => app.window.request_redraw(),
//!             Event::RedrawRequested(_window_id) => {
//!                 // (4) Call integration.begin_frame(), integration.end_frame(&mut window),
//!                 // integration.context().tessellate(shapes), integration.paint(...)
//!                 // in app.draw().
//!                 app.draw().unwrap();
//!             }
//!             _ => (),
//!         }
//!     })
//! }
//! // (5) Call integration.destroy() when drop app.
//! ```
//!
//! [Full example is in examples directory](https://github.com/MatchaChoco010/egui-winit-ash-integration/tree/main/examples)

mod allocator;
mod integration;

pub use allocator::*;
pub use integration::*;

#[cfg(feature = "gpu-allocator-feature")]
mod gpu_allocator;
#[cfg(feature = "gpu-allocator-feature")]
pub use crate::gpu_allocator::*;
