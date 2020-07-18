use crate::maths::Vec3;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub in_colour: [f32; 3],
}
