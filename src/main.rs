use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage, AttachmentImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::format::*;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError, Surface
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;

use specs::prelude::*;

mod maths;
use maths::{Vec3, Quaternion};

// A component contains data
// which is associated with an entity.
#[derive(Debug)]
struct Vel([f32;3]);

impl Component for Vel {
    type Storage = VecStorage<Self>;
}

#[derive(Debug)]
struct Pos ([f32;3]);

impl Component for Pos {
    type Storage = VecStorage<Self>;
}

#[derive(Debug)]
struct Rot (Quaternion);

impl Component for Rot {
	type Storage = VecStorage<Self>;
}

struct Draw {
	vertex_buffer: Arc<dyn vulkano::buffer::BufferAccess + Send + Sync>,
	index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
}

impl Component for Draw {
	type Storage = VecStorage<Self>;
}

struct SysA;

impl<'a> System<'a> for SysA {
    // These are the resources required for execution.
    // You can also define a struct and `#[derive(SystemData)]`,
    // see the `full` example.
    type SystemData = (WriteStorage<'a, Pos>, ReadStorage<'a, Vel>);

    fn run(&mut self, (mut pos, vel): Self::SystemData) {
        // The `.join()` combines multiple component storages,
        // so we get access to all entities which have
        // both a position and a velocity.
        for (pos, vel) in (&mut pos, &vel).join() {
            pos.0[0] += vel.0[0];
            pos.0[1] += vel.0[1];
            pos.0[2] += vel.0[2];
        }
    }
}


#[derive(Default, Debug, Clone)]
struct Vertex {
	position: Vec3,
	in_colour: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, in_colour);


mod vs {
	vulkano_shaders::shader! {
		ty: "vertex",
		src: "
			#version 450	

			layout (location = 0) in vec3 position;
			layout (location = 1) in vec3 in_colour;

			layout (set = 0, binding = 0) uniform Data {
				mat4 model;
				mat4 view;
				mat4 projection;
			};

			layout (location = 0) out vec3 vert_colour;

			void main() {
				vert_colour = in_colour;
				gl_Position = projection * view * model * vec4(position, 1.0);

			}
		"
	}	
}

mod fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		src: "
			#version 450

			layout (location = 0) in vec3 vert_colour;

			layout (location = 0) out vec4 colour;


			void main() {

				colour = vec4(vert_colour, 1.0);

			}
		"
	}	
}

#[derive(Default)]
struct RecreateSwapchain(bool);

struct RenderSys {
	surface: Arc<Surface<Window>>,
	device: Arc<Device>,
	queue: Arc<Queue>,
	swapchain: Arc<Swapchain<Window>>,
	uniform_buffer: CpuBufferPool<vs::ty::Data>,
	render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
	pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
	dynamic_state: DynamicState,
	framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
	projection: [[f32;4];4],
	previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl RenderSys {
	fn new(event_loop: &EventLoop<()>) -> RenderSys {
		
		let required_extensions = vulkano_win::required_extensions();

		let instance = Instance::new(None, &required_extensions, None).unwrap();

		let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

		println!(
			"Using device: {} (type: {:?})",
			physical.name(),
			physical.ty()
		);

		let surface = WindowBuilder::new()
			.build_vk_surface(&event_loop, instance.clone())
			.unwrap();


		let queue_family = physical
			.queue_families()
			.find(|&q| {
				q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
			})
			.unwrap();

	    
		let device_ext = DeviceExtensions {
			khr_swapchain: true,
			..DeviceExtensions::none()
		};

		let (device, mut queues) = Device::new(
			physical,
			physical.supported_features(),
			&device_ext,
			[(queue_family, 0.5)].iter().cloned(),
		)
		.unwrap();
	
		let queue = queues.next().unwrap();

		let (swapchain, images) = {

			let caps = surface.capabilities(physical).unwrap();

			let alpha = caps.supported_composite_alpha.iter().next().unwrap();

			let format = caps.supported_formats[0].0;


			let dimensions: [u32; 2] = surface.window().inner_size().into();

			Swapchain::new(
				device.clone(),
				surface.clone(),
				caps.min_image_count,
				format,
				dimensions,
				1,
				ImageUsage::color_attachment(),
				&queue,
				SurfaceTransform::Identity,
				alpha,
				PresentMode::Fifo,
				FullscreenExclusive::Default,
				true,
				ColorSpace::SrgbNonLinear,
			)
			.unwrap()
		};

		
		let vs = vs::Shader::load(device.clone()).unwrap();
		let fs = fs::Shader::load(device.clone()).unwrap();

		let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

		let render_pass = Arc::new(
			vulkano::single_pass_renderpass!(
				device.clone(),
				attachments: {
					color: {
						load: Clear,
						store: Store,
						format: swapchain.format(),
						samples: 1,
					},
					depth: {
						load: Clear,
						store: Store,
						format: Format::D16Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {depth}
				}
			)
			.unwrap()
		);

		let	pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<Vertex>()
				.vertex_shader(vs.main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(fs.main_entry_point(),	())
				.depth_stencil_simple_depth()
				.render_pass(Subpass::from(render_pass.clone(),	0).unwrap())
				.build(device.clone())
				.unwrap(),
		);

		let	mut	dynamic_state =	DynamicState {
			line_width:	None,
			viewports: None,
			scissors: None,
			compare_mask: None,
			write_mask:	None,
			reference: None,
		};

		let	(framebuffers, projection) =
			window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);

		let previous_frame_end = Some(sync::now(device.clone()).boxed());
		
		RenderSys {
			surface: surface,
			device: device,
			queue: queue,
			swapchain: swapchain,
			uniform_buffer: uniform_buffer,
			render_pass: render_pass,
			pipeline: pipeline,
			dynamic_state: dynamic_state,
			framebuffers: framebuffers,
			projection: projection,
			previous_frame_end: previous_frame_end,
		}
	}
}

impl<'a> System<'a> for RenderSys {
	type SystemData = (
		Write<'a, RecreateSwapchain>,
		ReadStorage<'a, Draw>,
		ReadStorage<'a, Pos>,
	);

	fn run(&mut self, data: Self::SystemData) {
		let (mut recreate_swapchain, draw, pos) = data;
		self.previous_frame_end.as_mut().unwrap().cleanup_finished();
		
		if recreate_swapchain.0 {
			println!("Recreating swapchain!");
			let	dimensions:	[u32; 2] = self.surface.window().inner_size().into();
			let	(new_swapchain,	new_images)	=
				match self.swapchain.recreate_with_dimensions(dimensions) {
					Ok(r) => r,
					Err(SwapchainCreationError::UnsupportedDimensions) => return,
					Err(e) => panic!("Failed to	recreate swapchain:	{:?}", e),
				};

			self.swapchain = new_swapchain;
			let (framebuffers, projection) = window_size_dependent_setup(
				self.device.clone(),
				&new_images,
				self.render_pass.clone(),
				&mut self.dynamic_state,
			);
			self.framebuffers = framebuffers;
			self.projection = projection;
			recreate_swapchain.0 = false;
		}

		let	(image_num,	suboptimal,	acquire_future)	=
			match swapchain::acquire_next_image(self.swapchain.clone(), None) {
				Ok(r) => r,
				Err(AcquireError::OutOfDate) =>	{
					recreate_swapchain.0 = true;
					return;
				}
				Err(e) => panic!("Failed to	acquire	next image:	{:?}", e),
			};

		if suboptimal {
			recreate_swapchain.0 = true;
		}

		let	clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

		let	mut	builder	= AutoCommandBufferBuilder::primary_one_time_submit(
			self.device.clone(),
			self.queue.family(),
		)
		.unwrap();

//		let mut to_draw = Vec::<Arc<dyn vulkano::buffer::BufferAccess + Send + Sync>>::new();

		builder

			.begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values)
			.unwrap();

		for (draw, pos) in (&draw, &pos).join() {



			let sub_buffer = {
				let uniform_data = vs::ty::Data {
					model: [
						[1.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0],
						[pos.0[0], pos.0[1], pos.0[2], 1.0],
					],
					view: [
						[1.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0],
						[0.0, 0.0, 0.0, 1.0],
					],
					projection: self.projection,
				};

				self.uniform_buffer.next(uniform_data).unwrap()
			};

		
			let set = Arc::new(vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
				self.pipeline.descriptor_set_layout(0).unwrap().clone())
				.add_buffer(sub_buffer).unwrap()
				.build().unwrap()
			);
		
			builder
				.draw_indexed(
					self.pipeline.clone(),
					&self.dynamic_state,
					vec!(draw.vertex_buffer.clone()),
					draw.index_buffer.clone(),
					set.clone(),
					(),
				)
				.unwrap();
		}
		builder
			.end_render_pass()
			.unwrap();

		let	command_buffer = builder.build().unwrap();

		let	future = self.previous_frame_end
			.take()
			.unwrap()
			.join(acquire_future)
			.then_execute(self.queue.clone(), command_buffer)
			.unwrap()
			.then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
			.then_signal_fence_and_flush();

		match future {
			Ok(future) => {
				self.previous_frame_end = Some(future.boxed());
			}
			Err(FlushError::OutOfDate) => {
				recreate_swapchain.0 = true;
				self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
			}
			Err(e) => {
				println!("Failed to	flush future: {:?}", e);
				self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
			}
		}
	}
}

fn main() {
	let event_loop = EventLoop::new();
	let render_sys = RenderSys::new(&event_loop);

	let vertex_buffer = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				Vertex {
					position: Vec3(-0.5, -0.5, 0.0),
					in_colour: [1.0, 0.0, 0.0],
				},
				Vertex {
					position: Vec3(0.0, 0.5, 0.0),
					in_colour: [0.0, 1.0, 0.0],
				},
				Vertex {
					position: Vec3(0.5, -0.5, 0.0),
					in_colour: [0.0, 0.0, 1.0],
				},
			]
			.iter()
			.cloned(),
		)
		.unwrap()
	};

	let vertex_buffer_1 = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				Vertex {
					position: Vec3(0.5, 0.5, 0.0),
					in_colour: [1.0, 0.0, 0.0],
				},
				Vertex {
					position: Vec3(-0.5, -0.0, 0.0),
					in_colour: [0.0, 1.0, 0.0],
				},
				Vertex {
					position: Vec3(0.5, -0.5, 0.0),
					in_colour: [0.0, 0.0, 1.0],
				},
			]
			.iter()
			.cloned(),
		)
		.unwrap()
	};

	let vertex_buffer_2 = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				Vertex {
					position: Vec3(1.0, 1.0, 1.0),
					in_colour: [1.0, 1.0, 1.0],
				},
				Vertex {
					position: Vec3(1.0, -1.0, 1.0),
					in_colour: [1.0, 0.0, 1.0],
				},
				Vertex {
					position: Vec3(-1.0, -1.0, 1.0),
					in_colour: [0.0, 0.0, 1.0],
				},
				Vertex {
					position: Vec3(-1.0, 1.0, 1.0),
					in_colour: [0.0, 1.0, 1.0],
				},
				Vertex {
					position: Vec3(1.0, 1.0, -1.0),
					in_colour: [1.0, 1.0, 0.0],
				},
				Vertex {
					position: Vec3(1.0, -1.0, -1.0),
					in_colour: [1.0, 0.0, 0.0],
				},
				Vertex {
					position: Vec3(-1.0, -1.0, -1.0),
					in_colour: [0.0, 0.0, 0.0],
				},
				Vertex {
					position: Vec3(-1.0, 1.0, -1.0),
					in_colour: [0.0, 1.0, 0.0],
				},
			]
			.iter()
			.cloned(),
		)
		.unwrap()
	};

	let index_buffer = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				0, 1, 2
			]
			.iter()
			.cloned()
		)
		.unwrap()
	};

	let index_buffer_1 = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				0, 1, 2
			]
			.iter()
			.cloned()
		)
		.unwrap()
	};

	let index_buffer_2 = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			[
				0, 1, 2, 2, 3, 0
			]
			.iter()
			.cloned()
		)
		.unwrap()
	};	

	let mut world = World::new();
	world.register::<Pos>();
	world.register::<Vel>();
	world.register::<Draw>();


	world.create_entity().with(Vel([0.001;3])).with(Pos([0.0, 1.0, -5.0])).with(Draw {vertex_buffer: vertex_buffer, index_buffer: index_buffer}).build();
	world.create_entity().with(Vel([-0.002;3])).with(Pos([-2.0, 0.0, -6.0])).with(Draw {vertex_buffer: vertex_buffer_1, index_buffer: index_buffer_1}).build();
	world.create_entity().with(Vel([0.003;3])).with(Pos([1.0, 3.0, -7.0])).with(Draw {vertex_buffer: vertex_buffer_2, index_buffer: index_buffer_2}).build();

	world.insert(RecreateSwapchain(false));

	let mut dispatcher = DispatcherBuilder::new()
		.with(SysA, "sys_a", &[])
		.with_thread_local(render_sys)
		.build();

	dispatcher.setup(&mut world);

	dispatcher.dispatch(&mut world);

	event_loop.run(move |event, _, control_flow| {
		match event	{
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => {
				*control_flow =	ControlFlow::Exit;
			}
			Event::WindowEvent {
				event: WindowEvent::Resized(_),
				..
			} => {
				println!("Should recreate swapchain");
				world.write_resource::<RecreateSwapchain>().0 = true;
			}
			Event::RedrawEventsCleared => {
				dispatcher.dispatch(&mut world);

			}
			_ => (),
		}
	});
}

static NEAR: f32 = 0.1;
static FAR: f32 = 1000.0;

fn create_projection_matrix(dimensions: (f32, f32)) -> [[f32;4];4] {
	println!("Creating projection matrix!");
	let aspect_ratio = dimensions.0 / dimensions.1;
	let y_scale = 1.0 / (90.0f32 / 2.0f32).to_radians().tan() * aspect_ratio;
	let x_scale = y_scale / aspect_ratio;
	let frustum_length = FAR - NEAR;
	let zp = FAR + NEAR;
	[
		[x_scale, 0.0, 0.0, 0.0],
		[0.0, y_scale, 0.0, 0.0],
		[0.0, 0.0, -frustum_length / zp, -1.0],
		[0.0, 0.0, -(2.0 * NEAR * FAR ) / frustum_length, 0.0]
	]
}

fn window_size_dependent_setup(
	device: Arc<Device>,
	images:	&[Arc<SwapchainImage<Window>>],
	render_pass: Arc<dyn RenderPassAbstract	+ Send + Sync>,
	dynamic_state: &mut	DynamicState,
) -> (Vec<Arc<dyn FramebufferAbstract + Send	+ Sync>>, [[f32;4];4]) {
	let	dimensions = images[0].dimensions();

	let	viewport = Viewport	{
		origin:	[0.0, 0.0],
		dimensions:	[dimensions[0] as f32, dimensions[1] as	f32],
		depth_range: 0.0..1.0,
	};
	dynamic_state.viewports	= Some(vec![viewport]);


	let depth_buffer =
		AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

	(images
		.iter()
		.map(|image| {
			Arc::new(
				Framebuffer::start(render_pass.clone())
					.add(image.clone())
					.unwrap()
					.add(depth_buffer.clone())
					.unwrap()
					.build()
					.unwrap(),
			) as Arc<dyn FramebufferAbstract + Send	+ Sync>
		})
		.collect::<Vec<_>>(), create_projection_matrix((dimensions[0] as f32, dimensions[1] as f32)))
}
