# egui-winit-ash-integration

[![Latest version](https://img.shields.io/crates/v/egui-winit-ash-integration.svg)](https://crates.io/crates/egui-winit-ash-integration)
[![Documentation](https://docs.rs/egui-winit-ash-integration/badge.svg)](https://docs.rs/egui-winit-ash-integration)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)
[![egui version: 0.14.2](https://img.shields.io/badge/egui%20version-0.14.2-orange)](https://docs.rs/egui/0.14.2/egui/index.html)

This is the [egui](https://github.com/emilk/egui) integration crate for [winit](https://github.com/rust-windowing/winit) and [ash](https://github.com/MaikKlein/ash).
The default GPU allocator is [gpu_allocator](https://github.com/Traverse-Research/gpu-allocator), but you can also implement AllocatorTrait.

# Example

```sh
cargo run --example example
```

```sh
cargo run --example user_texture
```

# Usage

```rust
fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    // (1) Call Integration::<Arc<Mutex<Allocator>>>::new() in App::new().
    let mut app = App::new(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        // (2) Call integration.handle_event(&event).
        app.egui_integration.handle_event(&event);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                // (3) Call integration.recreate_swapchain(...) in app.recreate_swapchain().
                app.recreate_swapchain().unwrap();
            }
            Event::MainEventsCleared => app.window.request_redraw(),
            Event::RedrawRequested(_window_id) => {
                // (4) Call integration.begin_frame(), integration.end_frame(&mut window),
                // integration.context().tessellate(shapes), integration.paint(...)
                // in app.draw().
                app.draw().unwrap();
            }
            _ => (),
        }
    })
}
// (5) Call integration.destroy() when drop app.
```

[Full example is in examples directory](https://github.com/MatchaChoco010/egui-winit-ash-integration/tree/main/examples)

# License

MIT OR Apache-2.0
