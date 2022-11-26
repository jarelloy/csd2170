#version 450

layout (binding = 0) uniform UBO 
{
	float tessLevel;
} ubo; 
 
layout (vertices = 3) out;
  
void main()
{
	if (gl_InvocationID == 0)
	{
		gl_TessLevelInner[0] = ubo.tessLevel;
		gl_TessLevelInner[1] = ubo.tessLevel;

		gl_TessLevelOuter[0] = ubo.tessLevel;
		gl_TessLevelOuter[1] = ubo.tessLevel;
		gl_TessLevelOuter[2] = ubo.tessLevel;		
		gl_TessLevelOuter[3] = ubo.tessLevel;		
	}
} 
