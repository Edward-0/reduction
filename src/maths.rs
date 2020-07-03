use vulkano::pipeline::vertex::VertexMember;
use vulkano::pipeline::vertex::VertexMemberTy;

#[derive(Default, Debug, Clone)]
pub struct Vec3(pub f32, pub f32, pub f32);

unsafe impl VertexMember for Vec3 {
	fn format() -> (VertexMemberTy, usize) {
		let (ty, sz) = <f32 as VertexMember>::format();
		(ty, sz * 3)
	}
}

impl Vec3 {
	fn dot(&self, other: Self) -> f32 {
		(self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2)
	}

	fn cross(&self, other: Self) -> Self {
		Vec3(
			self.1 * other.2 - self.2 - other.1,
			self.2 * other.0 - self.0 - other.2, 
			self.0 * other.1 - self.1 - other.0,)
	}
}

pub struct Mat4(
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
);

pub static IDENTITY_MAT4: Mat4 = Mat4(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, 0.0, 0.0, 1.0
);

unsafe impl VertexMember for Mat4 {
	fn format() -> (VertexMemberTy, usize) {
		let (ty, sz) = <f32 as VertexMember>::format();
		(ty, sz * 16)
	}
}

/*impl std::ops::Mul for Mat4 {
	type Output = Mat4;

	fn mul(self, other: Self) -> Mat4 {
		
	}
}**/

#[derive(Default, Debug, Clone)]
pub struct Vec4(f32, f32, f32, f32);

impl Vec4 {
	fn dot(&self, other: Self) -> f32 {
		(self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2) + (self.3 * other.3)
	}
}

#[derive(Default, Debug, Clone)]
pub struct Quaternion {
	s: f32,
	v: Vec3,
}
