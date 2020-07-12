# reduction
Simple rendering engine using Specs and Vulkan (vulkano). Includes a basic PLY model loader that can handle calculating normals.

## Implements
 * SLERP of Quaternions
 * PBR (based on Joey de Vries code from [Learn OpenGL](https://learnopengl.com)
 * Triangulation of abitrary polygons from PLY files
 * Vertex normal estimation
 
## Getting started

```sh
git clone "https://github.com/Edward-0/reduction"
cd reduction
cargo run
```
