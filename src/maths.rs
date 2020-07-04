use vulkano::pipeline::vertex::VertexMember;
use vulkano::pipeline::vertex::VertexMemberTy;
#[derive(Default, Debug, Clone, Copy)]
pub struct Vec3(pub f32, pub f32, pub f32);

unsafe impl VertexMember for Vec3 {
	fn format() -> (VertexMemberTy, usize) {
		let (ty, sz) = <f32 as VertexMember>::format();
		(ty, sz * 3)
	}
}

impl Vec3 {
	pub fn dot(self, other: Self) -> f32 {
		(self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2)
	}

	pub fn cross(self, other: Self) -> Self {
		Vec3(
			self.1 * other.2 - self.2 - other.1,
			self.2 * other.0 - self.0 - other.2, 
			self.0 * other.1 - self.1 - other.0,)
	}

	pub fn length(self) -> f32 {
		(self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt()
	}

	pub fn normalize(self) -> Vec3 {
		let l = self.length();
		Vec3(self.0 / l, self.1 / l, self.2 / l)
	}
}

impl std::ops::Mul<Vec3> for f32 {
	type Output = Vec3;
	
	fn mul(self, other: Vec3) -> Vec3 {
		Vec3(self * other.0, self * other.1, self * other.2)
	} 
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Mat4(
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
	pub f32, pub f32, pub f32, pub f32,
);

impl Mat4 {
	pub fn identity() -> Mat4 {
		IDENTITY_MAT4
	}

	pub fn projection(fov: f32, dimensions: (f32, f32), near: f32, far: f32) -> Mat4 {
		let aspect_ratio = dimensions.0 / dimensions.1;
		let y_scale = 1.0 / (fov / 2.0f32).tan() * aspect_ratio;
		let x_scale = y_scale / aspect_ratio;
		let frustum_length = far - near;
		let zp = far + near;
		Mat4 (
			x_scale, 0.0, 0.0, 0.0,
			0.0, y_scale, 0.0, 0.0,
			0.0, 0.0, -frustum_length / zp, -(2.0 * near * far ) / frustum_length,
			0.0, 0.0, -1.0, 0.0
		)
	}

	pub fn translate(self, amount: Vec3) -> Mat4 {
		self * Mat4 (
			1.0, 0.0, 0.0, amount.0,
			0.0, 1.0, 0.0, amount.1,
			0.0, 0.0, 1.0, amount.2,
			0.0, 0.0, 0.0, 1.0,
		)
	}

	pub fn as_array(self) -> [[f32;4];4] {
		[
			[self.0, self.1, self.2, self.3],
			[self.4, self.5, self.6, self.7],
			[self.8, self.9, self.10,self.11],
			[self.12,self.13,self.14,self.15],
		]
	}

	pub fn transpose(self) -> Self {
		Mat4(
			self.0, self.4, self.8, self.12,
			self.1, self.5, self.9, self.13,
			self.2, self.6, self.10,self.14,
			self.3, self.7, self.11,self.15
		)
	}
}

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

impl std::ops::Mul for Mat4 {
	type Output = Mat4;

	fn mul(self, other: Self) -> Self {
		Mat4(
			self.0 * other.0 + self.1 * other.4 + self.2 * other.8 + self.3 * other.12,
			self.0 * other.1 + self.1 * other.5 + self.2 * other.9 + self.3 * other.13,
			self.0 * other.2 + self.1 * other.6 + self.2 * other.10+ self.3 * other.14,
			self.0 * other.3 + self.1 * other.7 + self.2 * other.11+ self.3 * other.15,
			
			self.4 * other.0 + self.5 * other.4 + self.6 * other.8 + self.7 * other.12,
			self.4 * other.1 + self.5 * other.5 + self.6 * other.9 + self.7 * other.13,
			self.4 * other.2 + self.5 * other.6 + self.6 * other.10+ self.7 * other.14,
			self.4 * other.3 + self.5 * other.7 + self.6 * other.11+ self.7 * other.15,
		
			self.8 * other.0 + self.9 * other.4 + self.10* other.8 + self.11* other.12,
			self.8 * other.1 + self.9 * other.5 + self.10* other.9 + self.11* other.13,
			self.8 * other.2 + self.9 * other.6 + self.10* other.10+ self.11* other.14,
			self.8 * other.3 + self.9 * other.7 + self.10* other.11+ self.11* other.15,
		
			self.12* other.0 + self.13* other.4 + self.14* other.8 + self.15* other.12,
			self.12* other.1 + self.13* other.5 + self.14* other.9 + self.15* other.13,
			self.12* other.2 + self.13* other.6 + self.14* other.10+ self.15* other.14,
			self.12* other.3 + self.13* other.7 + self.14* other.11+ self.15* other.15,
		)
	}
}

#[derive(Default, Debug, Clone)]
pub struct Vec4(f32, f32, f32, f32);

impl Vec4 {
	fn dot(self, other: Self) -> f32 {
		(self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2) + (self.3 * other.3)
	}
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Quaternion {
	s: f32,
	v: Vec3,
}

impl Quaternion {

	// Expect unit axis
	pub fn rotation(axis: Vec3, theta: f32) -> Quaternion {
		Quaternion {
			s: (0.5 * theta).cos(),
			v: (0.5 * theta).sin() * axis,
		}
	}

	pub fn as_matrix(self) -> Mat4 {
		Mat4(
			1.0 - 2.0 * (self.v.1 * self.v.1 + self.v.2 * self.v.2),
			      2.0 * (self.v.0 * self.v.1 - self.v.2 * self.s),
			      2.0 * (self.v.0 * self.v.2 + self.v.1 * self.s),
			0.0,

			      2.0 * (self.v.0 * self.v.1 + self.v.2 * self.s),
			1.0 - 2.0 * (self.v.0 * self.v.0 + self.v.2 * self.v.2),
			      2.0 * (self.v.1 * self.v.2 - self.v.0 * self.s),
			0.0,
	
			      2.0 * (self.v.0 * self.v.2 - self.v.1 * self.s),
			      2.0 * (self.v.1 * self.v.2 + self.v.0 * self.s),
			1.0 - 2.0 * (self.v.0 * self.v.0 + self.v.1 * self.v.1),
			0.0,

			0.0,
			0.0,
			0.0,
			1.0,
		)
	}
}

impl std::ops::Mul for Quaternion {
	type Output = Self;

	fn mul(self, other: Self) -> Self {
		Quaternion {
			s: self.s * other.s - self.v.0 * other.v.0 - self.v.1 * other.v.1 - self.v.2 * other.v.2,
			v: Vec3 (
				self.s * other.v.0 + self.v.0 * other.s + self.v.1 * other.v.2 - self.v.2 * other.v.1,
				self.s * other.v.1 + self.v.1 * other.s + self.v.2 * other.v.0 - self.v.0 * other.v.2,
				self.s * other.v.2 + self.v.2 * other.s + self.v.0 * other.v.1 - self.v.1 * other.v.0,
			)
		}
	}
}

