#version 450	

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 in_colour;

layout (set = 0, binding = 0) uniform Data {
	mat4 model;
	mat4 view;
	mat4 projection;
};

layout (location = 0) out vec3 vert_colour;
layout (location = 1) out vec4 vert_z;
layout (location = 2) out vec3 vert_normal;

void main() {
	vec4 world_position = model * vec4(position, 1.0);
	vert_colour = in_colour;
	gl_Position = projection * view * world_position;
	vert_z = gl_Position;
	vert_normal = mat3(transpose(inverse(model))) * normal;
//	vert_normal = mat3(model) * normal;
}

