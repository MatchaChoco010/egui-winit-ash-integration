#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outViewPos;
layout(location = 1) out vec3 outViewNormal;

layout(binding = 0, set = 0) uniform UniformBufferObject {
  vec3 light;
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

void main() {
  mat4 mvp = ubo.proj * ubo.view * ubo.model;
  gl_Position = mvp * vec4(inPos, 1.0);
  outViewPos = ubo.view * ubo.model * vec4(inPos, 1.0);
  mat3 normalModelToView = transpose(inverse(mat3(ubo.view * ubo.model)));
  outViewNormal = normalModelToView * inNormal;
}
