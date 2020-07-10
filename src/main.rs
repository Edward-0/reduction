#![feature(thread_id_value)]
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

use serde::{Serialize, Deserialize};

use std::thread;

use std::time::Instant;

use std::sync::Arc;

use specs::prelude::*;

extern crate packed_simd;

mod maths;
use maths::{Vec3, Quaternion, Mat4};

mod vertex;
use vertex::Vertex;

mod ply;
use ply::StanfordPLY;
// A component contains data
// which is associated with an entity.
#[derive(Debug)]
struct Vel([f32;3]);

impl Component for Vel {
    type Storage = VecStorage<Self>;
}

#[derive(Debug)]
struct Pos (Vec3);

impl Component for Pos {
    type Storage = VecStorage<Self>;
}

#[derive(Debug)]
struct Scale (Vec3);

impl Component for Scale {
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


vulkano::impl_vertex!(Vertex, position, in_colour);


struct RotSys;

impl<'a> System<'a> for RotSys {
	type SystemData = (Read<'a, FrameNumber>, WriteStorage<'a, Rot>);

	fn run(&mut self, data: Self::SystemData) {
		let (frame_number, mut rot) = data;
	
		let q0 = Quaternion::rotation(Vec3(0.0, -1.0, 1.0).normalize(), 90.0f32.to_radians());
		let q1 = Quaternion::rotation(Vec3(1.0, 1.0, 0.0).normalize(), 270.0f32.to_radians());
		let q2 = Quaternion::rotation(Vec3(1.0, 1.0, 1.0).normalize(), 90.0f32.to_radians());
		let de = frame_number.0 as f32 / 256.0;
//		let de = 1.0f32;

		//let de = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32].iter().map(|x| (de * x * 2.0 * std::f32::consts::PI).sin() * x.recip()).sum();
//		println!("{}", de);
		for rot in (&mut rot).join() {
		rot.0 = q0.slerp(q1, (de * std::f32::consts::PI).sin()).slerp(q2, (de * std::f32::consts::PI * 4.0f32).cos());
		}
	}
}

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
			layout (location = 1) out vec4 vert_z;

			void main() {
				vert_colour = in_colour;
				gl_Position = projection * view * model * vec4(position, 1.0);
				vert_z = gl_Position;

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
			layout (location = 1) in vec4 vert_z;

			layout (location = 0) out vec4 colour;

			void main() {

				colour = vec4(vert_colour, 1.0);

			}
		"
	}	
}

struct FrameNumber(u64, Instant);

impl Default for FrameNumber {
	fn default() -> Self {
		Self(u64::default(), Instant::now())
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
	projection: Mat4,
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
						format: Format::D32Sfloat,
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
		Write<'a, FrameNumber>,
		ReadStorage<'a, Draw>,
		ReadStorage<'a, Pos>,
		ReadStorage<'a, Rot>,
		ReadStorage<'a, Scale>,
	);

	fn run(&mut self, data: Self::SystemData) {
		let (mut recreate_swapchain, mut frame_number, draw, pos, rot, scale) = data;
		self.previous_frame_end.as_mut().unwrap().cleanup_finished();
		
		frame_number.0 += 1;

		let elapsed = frame_number.1.elapsed().as_secs();

		if elapsed != 0 && frame_number.0 % 256 == 0 {
			println!("FPS: {}", frame_number.0 / elapsed);
		}

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

		for (draw, pos, rot, scale) in (&draw, &pos, &rot, &scale).join() {



			let sub_buffer = {
				let uniform_data = vs::ty::Data {
					model: (Mat4::identity()
						.translate(pos.0)
						* rot.0.as_matrix())
						.scale(scale.0)
						.transpose()
						.as_array(),
					view: [
						[1.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0],
						[0.0, 0.0, 0.0, 1.0],
					],
					projection: self.projection.transpose().as_array(),
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

//		println!("{}", thread::current().id().as_u64());
	
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


use std::boxed::Box;
use std::error::Error;

fn main() {

	//println!("{:#?}", gltf.buffers().next().unwrap().length());

	//println!("{:#?}", gltf);

	let ply_ply = StanfordPLY::new(include_str!("../../dragon_recon/dragon_vrip.ply").to_string());
//	let ply_ply = StanfordPLY::new(include_str!("../untitled.ply").to_string());
//	let ply_ply = StanfordPLY::new(include_str!("../bunny/reconstruction/bun_zipper.ply").to_string());

//	println!("{:#?}", ply_ply.vertices());
//	println!("{:#?}", ply_ply.indices());

	//println!(include_str!("../untitled.ply"));
	let event_loop = EventLoop::new();
	let render_sys = RenderSys::new(&event_loop);

	let vertex_buffer = {
		CpuAccessibleBuffer::from_iter(
			render_sys.device.clone(),
			BufferUsage::all(),
			false,
			ply_ply.vertices()
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
			ply_ply.indices()
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
				0, 1, 2, 2, 3, 0,
				4, 5, 6, 6, 7, 4,
				
				0, 1, 4, 4, 5, 1,
				2, 3, 6, 6, 7, 3,

				0, 3, 4, 4, 7, 3,
				5, 6, 2, 2, 1, 5
			]
			.iter()
			.cloned()
		)
		.unwrap()
	};	

	let mut world = World::new();
	world.register::<Pos>();
	world.register::<Vel>();
	world.register::<Rot>();
	world.register::<Draw>();
	world.register::<Scale>();


	world.create_entity()
		.with(Vel([0.001;3])).with(Pos(Vec3(0.0, 1.0, -5.0))).with(Scale(Vec3(10.0, 10.0, 10.0)))
		.with(Rot(Quaternion::rotation(Vec3(1.0, 1.0, 0.0).normalize(), 90f32.to_radians())))
		.with(Draw {vertex_buffer: vertex_buffer.clone(), index_buffer: index_buffer.clone()}).build();
	world.create_entity()
		.with(Vel([0.002;3])).with(Pos(Vec3(-2.0, 0.0, -6.0))).with(Scale(Vec3(1.0, 1.0, 1.0)))
		.with(Rot(Quaternion::rotation(Vec3(1.0, 1.0, 0.0).normalize(), 90f32.to_radians())))
		.with(Draw {vertex_buffer: vertex_buffer.clone(), index_buffer: index_buffer.clone()}).build();
	world.create_entity()
		.with(Vel([0.003;3])).with(Pos(Vec3(0.0, 0.0, -6.0))).with(Scale(Vec3(1.0, 1.0, 1.0)))
		.with(Rot(Quaternion::rotation(Vec3(1.0, 1.0, 0.0).normalize(),90f32.to_radians())))
		.with(Draw {vertex_buffer: vertex_buffer, index_buffer: index_buffer}).build();

	world.insert(RecreateSwapchain(false));

	world.insert(FrameNumber(0, Instant::now()));

	let mut dispatcher = DispatcherBuilder::new()
		.with(RotSys, "rot_sys", &[])
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

fn window_size_dependent_setup(
	device: Arc<Device>,
	images:	&[Arc<SwapchainImage<Window>>],
	render_pass: Arc<dyn RenderPassAbstract	+ Send + Sync>,
	dynamic_state: &mut	DynamicState,
) -> (Vec<Arc<dyn FramebufferAbstract + Send	+ Sync>>, Mat4) {
	let	dimensions = images[0].dimensions();

	let	viewport = Viewport	{
		origin:	[0.0, 0.0],
		dimensions:	[dimensions[0] as f32, dimensions[1] as	f32],
		depth_range: 0.0..1.0,
	};
	dynamic_state.viewports	= Some(vec![viewport]);


	let depth_buffer =
		AttachmentImage::transient(device.clone(), dimensions, Format::D32Sfloat).unwrap();

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
		.collect::<Vec<_>>(), Mat4::projection(70f32.to_radians(), (dimensions[0] as f32, dimensions[1] as f32), 0.001, 1000.0))
}
