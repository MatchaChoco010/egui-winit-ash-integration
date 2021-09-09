#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inViewPos;
layout(location = 1) in vec3 inViewNormal;

layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 0) uniform UniformBufferObject {
  vec3 light;
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

vec3 lightColor = vec3(1, 1, 1);
vec3 Kd = vec3(0.6, 0.6, 0.6);
vec3 Ks = vec3(0.6, 0.6, 0.6);
float shininess = 50;
vec3 ambient = vec3(0.3, 0.1, 0.1);

void main() {
  vec4 viewLightPos = ubo.view * vec4(ubo.light, 1.0);
  vec3 L = normalize(viewLightPos.xyz - inViewPos.xyz);
  vec3 V = normalize(-inViewPos.xyz);
  vec3 N = normalize(inViewNormal);
  vec3 H = normalize(L + V);

  vec3 diffuse = Kd * lightColor * max(dot(L, N), 0);

  vec3 specular = Ks * lightColor * pow(max(dot(N, H), 0), shininess);

  outColor = vec4(diffuse + specular + ambient, 1);
}
