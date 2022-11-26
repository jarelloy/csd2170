#version 450

layout (location = 1) in vec3 inPos;

layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor.rgb = inPos;
}