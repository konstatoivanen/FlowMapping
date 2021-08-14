#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <GLFW/glfw3.h>
#include <GLAD/glad.h>
#if _WIN32
#include <Windows.h>
#endif 

const static char* shader_vertex_texture =
"    #version 450 core                                      \n"
"                                                           \n"
"    layout(location = 0) in vec4 in_POSITION0;             \n"
"    out vec2 vs_TEXCOORD0;                                 \n"
"                                                           \n"
"    void main()                                            \n"
"    {                                                      \n"
"        gl_Position = in_POSITION0;                        \n"
"        vs_TEXCOORD0 = in_POSITION0.xy * 0.5f + 0.5f.xx;   \n"
"    };                                                     \n";

const static char* shader_vertex_boid =
"    #version 450 core                                                                                                      \n"
"                                                                                                                           \n"
"    #define PK_DECLARE_BUFFER(ValueType, BufferName) layout(std430) buffer BufferName { ValueType BufferName##_Data[]; }   \n"
"    #define PK_BUFFER_DATA(BufferName, index) BufferName##_Data[index]                                                     \n"
"                                                                                                                           \n"
"    PK_DECLARE_BUFFER(vec4, pk_Boids);                                                                                     \n"
"                                                                                                                           \n"
"    layout(location = 0) in vec4 in_POSITION0;                                                                             \n"
"                                                                                                                           \n"
"    out vec2 vs_TEXCOORD0;                                                                                                 \n"
"                                                                                                                           \n"
"    void main()                                                                                                            \n"
"    {                                                                                                                      \n"
"        vec4 position = in_POSITION0;                                                                                      \n"
"                                                                                                                           \n"
"        vs_TEXCOORD0 = in_POSITION0.xy * 0.5f;                                                                             \n"
"                                                                                                                           \n"
"        position.xy += 1.0f;                                                                                               \n"
"        position.xy *= 0.5f;                                                                                               \n"
"                                                                                                                           \n"
"        position.xy -= 0.5f;                                                                                               \n"
"        position.xy *= 0.01f;                                                                                              \n"
"        position.y *= 2.0f;                                                                                                \n"
"                                                                                                                           \n"
"        vec4 boid = PK_BUFFER_DATA(pk_Boids, gl_InstanceID);                                                               \n"
"                                                                                                                           \n"
"        position.xy += boid.xy;                                                                                            \n"
"                                                                                                                           \n"
"        position.xy -= 0.5f;                                                                                               \n"
"        position.xy *= 2.0f;                                                                                               \n"
"                                                                                                                           \n"
"        gl_Position = position;                                                                                            \n"
"    };                                                                                                                     \n";

const static char* shader_compute_iterate =
"   #version 450 core                                                                       \n"
"                                                                                           \n"
"   layout(r32ui) uniform uimage2D pk_FlowTexture;                                          \n"
"   uniform sampler2D pk_Maze;                                                              \n"
"   uniform ivec2 pk_CursorCoord;                                                           \n"
"                                                                                           \n"
"   layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;                      \n"
"   void main()                                                                             \n"
"   {                                                                                       \n"
"       ivec2 coord = ivec2(gl_GlobalInvocationID.xy);                                      \n"
"                                                                                           \n"
"       if (coord.x + coord.y == 0)                                                         \n"
"       {                                                                                   \n"
"           imageStore(pk_FlowTexture, pk_CursorCoord, uvec4(0xFFFFFF, 0, 0, 0));           \n"
"       }                                                                                   \n"
"                                                                                           \n"
"       if (texelFetch(pk_Maze, coord, 0).r > 0)                                            \n"
"       {                                                                                   \n"
"           imageStore(pk_FlowTexture, coord, 0u.xxxx);                                     \n"
"           return;                                                                         \n"
"       }                                                                                   \n"
"                                                                                           \n"
"       uint n0 = imageLoad(pk_FlowTexture, coord + ivec2(1,  0)).x;                        \n"
"       uint n1 = imageLoad(pk_FlowTexture, coord + ivec2(0,  1)).x;                        \n"
"       uint n2 = imageLoad(pk_FlowTexture, coord + ivec2(0, -1)).x;                        \n"
"       uint n3 = imageLoad(pk_FlowTexture, coord + ivec2(-1, 0)).x;                        \n"
"                                                                                           \n"
"       uint maxvalue = max(max(max(n0,n1),n2),n3);                                         \n"
"                                                                                           \n"
"       if (maxvalue == 0)                                                                  \n"
"       {                                                                                   \n"
"           return;                                                                         \n"
"       }                                                                                   \n"
"                                                                                           \n"
"       imageStore(pk_FlowTexture, coord, uvec4(max(maxvalue - 1, 0), 0, 0, 0));            \n"
"   }                                                                                       \n";

const static char* shader_compute_boids =
"   #version 450 core                                                                                                                       \n"
"                                                                                                                                           \n"
"   #define PK_DECLARE_BUFFER(ValueType, BufferName) layout(std430) buffer BufferName { ValueType BufferName##_Data[]; }                    \n"
"   #define PK_BUFFER_DATA(BufferName, index) BufferName##_Data[index]                                                                      \n"
"                                                                                                                                           \n"
"   float NoiseUV(float u, float v) { return fract(43758.5453 * sin(dot(vec2(12.9898, 78.233), vec2(u, v)))); }                             \n"
"                                                                                                                                           \n"
"   vec3 NoiseUV3(vec2 uv) { return vec3(NoiseUV(uv.x + 57.0f, uv.y), NoiseUV(uv.x, uv.y + 57.0f), NoiseUV(uv.x, uv.y)); }                  \n"
"                                                                                                                                           \n"
"   PK_DECLARE_BUFFER(vec4, pk_Boids);                                                                                                      \n"
"   layout(r32ui) uniform uimage2D pk_FlowTexture;                                                                                          \n"
"   uniform float pk_Time;                                                                                                                  \n"
"                                                                                                                                           \n"
"   vec2 GetTargetVelocity(ivec2 coord)                                                                                                     \n"
"   {                                                                                                                                       \n"
"       float x0 = imageLoad(pk_FlowTexture, coord + ivec2(1, 0)).x;                                                                        \n"
"       float x1 = imageLoad(pk_FlowTexture, coord + ivec2(-1, 0)).x;                                                                       \n"
"       float y0 = imageLoad(pk_FlowTexture, coord + ivec2(0, 1)).x;                                                                        \n"
"       float y1 = imageLoad(pk_FlowTexture, coord + ivec2(0, -1)).x;                                                                       \n"
"       vec2 direction = clamp(vec2(x0.x - x1.x, y0.x - y1.x), -1.0f.xx, 1.0f.xx);                                                          \n"
"       return direction;                                                                                                                   \n"
"   }                                                                                                                                       \n"
"                                                                                                                                           \n"
"   layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;                                                                       \n"
"   void main()                                                                                                                             \n"
"   {                                                                                                                                       \n"
"       vec2 size = imageSize(pk_FlowTexture).xy;                                                                                           \n"
"       vec4 boid = PK_BUFFER_DATA(pk_Boids, gl_GlobalInvocationID.x);                                                                      \n"
"       vec2 position = boid.xy;                                                                                                            \n"
"       float speed = boid.z;                                                                                                               \n"
"       float killtime = boid.w;                                                                                                            \n"
"                                                                                                                                           \n"
"       ivec2 coord = ivec2(position * size);                                                                                               \n"
"                                                                                                                                           \n"
"       if (pk_Time > killtime)                                                                                                             \n"
"       {                                                                                                                                   \n"
"           vec3 noise = NoiseUV3(vec2(coord + gl_GlobalInvocationID.xy));                                                                  \n"
"           position = vec2(noise.xy);                                                                                                      \n"
"                                                                                                                                           \n"
"           killtime = pk_Time + 60.0f + (1.0f - noise.z) * 120.0f;                                                                         \n"
"           speed = mix(0.25f, 1.0f, noise.z);                                                                                              \n"
"       }                                                                                                                                   \n"
"                                                                                                                                           \n"
"       coord = ivec2(position * size);                                                                                                     \n"
"                                                                                                                                           \n"
"       int xoffset = int(imageLoad(pk_FlowTexture, coord + ivec2(1, 0)).x) - int(imageLoad(pk_FlowTexture, coord + ivec2(-1, 0)).x);       \n"
"       int yoffset = int(imageLoad(pk_FlowTexture, coord + ivec2(0, 1)).x) - int(imageLoad(pk_FlowTexture, coord + ivec2(0, -1)).x);       \n"
"                                                                                                                                           \n"
"       position.xy += GetTargetVelocity(coord) * speed.xx / size.xx;                                                                       \n"
"       position.xy = clamp(position.xy, 0.0f, 1.0f);                                                                                       \n"
"                                                                                                                                           \n"
"       PK_BUFFER_DATA(pk_Boids, gl_GlobalInvocationID.x) = vec4(position, speed, killtime);                                                \n"
"   }                                                                                                                                       \n";

const static char* shader_fragment_texture =
"   #version 450 core                                                                           \n"
"                                                                                               \n"
"   layout(r32ui) uniform readonly uimage2D pk_FlowTexture;                                     \n"
"   uniform sampler2D pk_Maze;                                                                  \n"
"                                                                                               \n"
"   vec3 HSVToRGB(vec3 c)                                                                       \n"
"   {                                                                                           \n"
"       vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);                                          \n"
"       vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);                                       \n"
"       return c.zzz * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);                             \n"
"   }                                                                                           \n"
"                                                                                               \n"
"   vec3 HSVToRGB(float hue, float saturation, float value)                                     \n"
"   {                                                                                           \n"
"       return HSVToRGB(vec3(hue, saturation, value));                                          \n"
"   }                                                                                           \n"
"                                                                                               \n"
"   in vec2 vs_TEXCOORD0;                                                                       \n"
"   layout(location = 0) out vec4 SV_Target0;                                                   \n"
"                                                                                               \n"
"   void main()                                                                                 \n"
"   {                                                                                           \n"
"       ivec2 coord = ivec2(vs_TEXCOORD0 * imageSize(pk_FlowTexture).xy);                       \n"
"       uint value = imageLoad(pk_FlowTexture, coord).x;                                        \n"
"                                                                                               \n"
"       float gradient = 1.0f - clamp((0xFFFFFF - value.x) / 8192.0f, 0.0, 1.0f);               \n"
"       float H = gradient * 3.0f + 0.6f;                                                       \n"
"       float S = mix(0.0f, 0.8f, gradient);                                                    \n"
"       float V = mix(0.05f, 0.8f, gradient * gradient);                                        \n"
"                                                                                               \n"
"       vec4 color = 1.0f.xxxx;                                                                 \n"
"                                                                                               \n"
"       color.rgb = HSVToRGB(H,S,V);                                                            \n"
"                                                                                               \n"
"       vec4 offset = vec4(0.25f, 0.25f, -0.25f, -0.25f) / textureSize(pk_Maze, 0).xyxy;        \n"
"                                                                                               \n"
"       float border = 0.0f;                                                                    \n"
"       border += texture(pk_Maze, vs_TEXCOORD0 + offset.xy).r;                                 \n"
"       border += texture(pk_Maze, vs_TEXCOORD0 + offset.zw).r;                                 \n"
"       border += texture(pk_Maze, vs_TEXCOORD0 + offset.xw).r;                                 \n"
"       border += texture(pk_Maze, vs_TEXCOORD0 + offset.zy).r;                                 \n"
"       border = smoothstep(border, 0.9f, 0.25f);                                               \n"
"       border *= border;                                                                       \n"
"       border *= 0.75f;                                                                        \n"
"                                                                                               \n"
"       color.rgb = mix(color.rgb, 1.0f.xxx, border);                                           \n"
"                                                                                               \n"
"       SV_Target0 = color;                                                                     \n"
"   };                                                                                          \n";

const static char* shader_fragment_boid =
"   #version 450 core                               \n"
"                                                   \n"
"   in vec2 vs_TEXCOORD0;                           \n"
"                                                   \n"
"   layout(location = 0) out vec4 SV_Target0;       \n"
"                                                   \n"
"   void main()                                     \n"
"   {                                               \n"
"       if (length(vs_TEXCOORD0) > 0.5f)            \n"
"       {                                           \n"
"           discard;                                \n"
"       }                                           \n"
"                                                   \n"
"       SV_Target0 = vec4(1.0f, 0.0f, 0.0f, 1.0f);  \n"
"   };                                              \n";

const static char CELL_VISITED = 1 << 5;
const static int OFFSETSX[4] = { 1,  0, -1, 0 };
const static int OFFSETSY[4] = { 0, -1,  0, 1 };

struct ShaderSource
{
	GLenum type;
	const char* source;
};

static GLuint CompileShader(std::initializer_list<ShaderSource> sources)
{
	auto program = glCreateProgram();

	auto stageCount = sources.size();

	GLenum* glShaderIDs = reinterpret_cast<GLenum*>(alloca(sizeof(GLenum) * stageCount));

	int glShaderIDIndex = 0;

	for (auto& kv : sources)
	{
		auto type = kv.type;
		const auto& source = kv.source;

		auto glShader = glCreateShader(type);

		glShaderSource(glShader, 1, &source, 0);
		glCompileShader(glShader);

		GLint isCompiled = 0;
		glGetShaderiv(glShader, GL_COMPILE_STATUS, &isCompiled);

		if (isCompiled == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetShaderiv(glShader, GL_INFO_LOG_LENGTH, &maxLength);

			std::vector<GLchar> infoLog(maxLength);
			glGetShaderInfoLog(glShader, maxLength, &maxLength, infoLog.data());

			glDeleteShader(glShader);

			printf(infoLog.data());
			glDeleteShader(glShader);
			continue;
		}

		glAttachShader(program, glShader);
		glShaderIDs[glShaderIDIndex++] = glShader;
	}

	glLinkProgram(program);

	GLint isLinked = 0;
	glGetProgramiv(program, GL_LINK_STATUS, (int*)&isLinked);

	if (isLinked == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> infoLog(maxLength);
		glGetProgramInfoLog(program, maxLength, &maxLength, infoLog.data());

		glDeleteProgram(program);

		for (auto i = 0; i < stageCount; ++i)
		{
			glDeleteShader(glShaderIDs[i]);
		}

		printf("Shader link failure! \n%s", infoLog.data());
		glDeleteProgram(program);
		return 0;
	}

	for (auto i = 0; i < stageCount; ++i)
	{
		glDetachShader(program, glShaderIDs[i]);
		glDeleteShader(glShaderIDs[i]);
	}

	return program;
}

static void GenerateMaze(uint32_t w, uint32_t h, uint32_t pw, char* texture)
{
    auto tw = w * pw;
    auto th = h * pw;

    auto maze = reinterpret_cast<char*>(calloc(w * h, sizeof(char)));
    auto stack = reinterpret_cast<uint32_t*>(malloc(w * h * sizeof(uint32_t)));

    auto offs = pw - 1u;
    auto stackSize = 1u;
    auto ncells = w * h - 1u;
    maze[0] = CELL_VISITED;
    stack[0] = 0u;

    while (ncells > 0)
    {
        uint32_t i = stack[stackSize - 1];
        uint32_t n = 0u;
        uint32_t cx = i % w;
        uint32_t cy = i / w;

        auto dir = 0xFF;

        for (uint32_t j = rand(), k = j + 4; j < k; ++j)
        {
            int nx = cx + OFFSETSX[j % 4];
            int ny = cy + OFFSETSY[j % 4];
            n = nx + ny * w;

            if (nx >= 0 && nx < w && ny >= 0 && ny < h && (maze[n] & CELL_VISITED) == 0)
            {
                dir = j % 4;
                break;
            }
        }

        if (dir == 0xFF)
        {
            stackSize--;
            continue;
        }

        maze[n] |= CELL_VISITED | (1 << ((dir + 2) % 4));
        maze[i] |= (1 << dir);

        stack[stackSize++] = n;
        ncells--;
    }

    for (auto x = 0u; x < w; ++x)
    for (auto y = 0u; y < h; ++y)
    {
        auto cell = maze[x + y * w];
        auto tx = x * pw;
        auto ty = y * pw;

        if ((cell & (1 << 3)) == 0 && tx > 0)
        {
            texture[tx - 1 + (ty + offs) * tw] = 255;
        }

        for (auto i = 0; i < pw; ++i)
        {
            if ((cell & (1 << 0)) == 0)
            {
                texture[tx + offs + (ty + i) * tw] = 255;
            }

            if ((cell & (1 << 3)) == 0)
            {
                texture[tx + i + (ty + offs) * tw] = 255;
            }
        }
    }

    free(maze);
    free(stack);
}

int main(int argc, char* argv[])
{
	GLFWwindow* window = nullptr;

    #if _WIN32
        ::ShowWindow(::GetConsoleWindow(), SW_SHOW);
    #endif

    if (!glfwInit())
	{
		printf("Failed to initialize glfw!");
		return 0;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow((int)1024, (int)512, "Maze Path Finding", nullptr, nullptr);

	if (!window)
	{
		printf("Failed to create a window!");
		glfwTerminate();
		return 0;
	}

	glfwMakeContextCurrent(window);

	int gladstatus = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	if (gladstatus == 0)
	{
		printf("Failed to load glad!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return 0;
	}

    glfwSwapInterval(1);

    auto shaderDisplayTexture = CompileShader({ { GL_VERTEX_SHADER, shader_vertex_texture }, { GL_FRAGMENT_SHADER, shader_fragment_texture} });
    auto shaderDisplayBoids = CompileShader({ { GL_VERTEX_SHADER, shader_vertex_boid }, { GL_FRAGMENT_SHADER, shader_fragment_boid } });
    auto shaderComputeIterate = CompileShader({ { GL_COMPUTE_SHADER, shader_compute_iterate } });
    auto shaderComputeBoids = CompileShader({ { GL_COMPUTE_SHADER, shader_compute_boids } });

    auto graphWidth = 64;
    auto graphHeight = 32;
    auto graphPadding = 6;
    auto texWidth = graphWidth * graphPadding;
    auto texHeight = graphHeight * graphPadding;
    
    auto mazeTexture = reinterpret_cast<char*>(calloc(texWidth * texHeight, sizeof(char)));

    GenerateMaze(graphWidth, graphHeight, graphPadding, mazeTexture);

	GLuint textureIds[2];
    GLuint boidBufferId;
	glCreateTextures(GL_TEXTURE_2D, 2, textureIds);
	glTextureStorage2D(textureIds[0], 1, GL_R8, texWidth, texHeight);
	glTextureStorage2D(textureIds[1], 1, GL_R32UI, texWidth, texHeight);
    glTextureParameteri(textureIds[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(textureIds[1], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureSubImage2D(textureIds[0], 0, 0, 0, texWidth, texHeight, GL_RED, GL_UNSIGNED_BYTE, mazeTexture);
    free(mazeTexture);

    glBindTextureUnit(0, textureIds[0]);
    glBindImageTexture(0, textureIds[1], 0, false, 0, GL_READ_WRITE, GL_R32UI);

    glGenBuffers(1, &boidBufferId);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, boidBufferId);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, 1024 * 4 * sizeof(float), nullptr, GL_NONE);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boidBufferId);

    float vertices[] = 
    { 
        -1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f
    };

    unsigned int indices[] = { 0,1,2, 2,3,0 };

    GLuint vertexArrayObject;
	glGenVertexArrays(1, &vertexArrayObject);
	glBindVertexArray(vertexArrayObject);

    GLuint indexBufferId;
    glCreateBuffers(1, &indexBufferId);
    glBindBuffer(GL_ARRAY_BUFFER, indexBufferId);
    glBufferStorage(GL_ARRAY_BUFFER, 6 * sizeof(uint32_t), indices, GL_NONE);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferId);

    GLuint vertexBufferId;
    glCreateBuffers(1, &vertexBufferId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glBufferStorage(GL_ARRAY_BUFFER, 4 * 4 * sizeof(float), vertices, GL_NONE);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (const void*)0);

    glShaderStorageBlockBinding(shaderComputeBoids, 0, 0);
    glShaderStorageBlockBinding(shaderDisplayBoids, 0, 0);

    glUseProgram(shaderDisplayTexture);
	glUniform1i(glGetUniformLocation(shaderDisplayTexture, "pk_FlowTexture"), 0);
	glUniform1i(glGetUniformLocation(shaderDisplayTexture, "pk_Maze"), 0);

	glUseProgram(shaderComputeIterate);
    glUniform1i(glGetUniformLocation(shaderComputeIterate, "pk_FlowTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderComputeIterate, "pk_Maze"), 0);
    auto cursorCoordLocation = glGetUniformLocation(shaderComputeIterate, "pk_CursorCoord");

	glUseProgram(shaderComputeBoids);
    glUniform1i(glGetUniformLocation(shaderComputeBoids, "pk_FlowTexture"), 0);
    auto timeLocation = glGetUniformLocation(shaderComputeBoids, "pk_Time");

    glFrontFace(GL_CW);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        ypos /= height;
        xpos /= width;

        auto coordx = (uint32_t)(xpos * texWidth);
        auto coordy = (uint32_t)((1.0f - ypos) * texHeight);
        
        coordx = coordx < 0 ? 0 : coordx > texWidth ? texWidth - 1 : coordx;
        coordy = coordy < 0 ? 0 : coordy > texHeight ? texHeight - 1 : coordy;

        glUseProgram(shaderComputeIterate);
        
        if (glfwGetMouseButton(window, 0))
        {
            glUniform2i(cursorCoordLocation, coordx, coordy);
        }
        
        for (auto i = 0; i < 4; ++i)
        {
            glDispatchCompute(texWidth / 32, texHeight / 32, 1);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        
        glUseProgram(shaderComputeBoids);
        glUniform1f(timeLocation, glfwGetTime());
        glDispatchCompute(32, 1 , 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(shaderDisplayTexture);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)(size_t)0);

        glUseProgram(shaderDisplayBoids);
        glDrawElementsInstancedBaseInstance(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)(size_t)0, (GLsizei)1024, (GLuint)0);
        glfwSwapBuffers(window);
    }

    glDeleteVertexArrays(1, &vertexArrayObject);
    glDeleteBuffers(1, &boidBufferId);
    glDeleteBuffers(1, &indexBufferId);
    glDeleteBuffers(1, &vertexBufferId);
	glDeleteTextures(2, textureIds);
	glDeleteProgram(shaderDisplayTexture);
	glDeleteProgram(shaderDisplayBoids);
	glDeleteProgram(shaderComputeIterate);
	glDeleteProgram(shaderComputeBoids);
	glfwDestroyWindow(window);
	glfwTerminate();
    return 0;
}
