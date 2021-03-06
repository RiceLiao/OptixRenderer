/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 //-----------------------------------------------------------------------------
 //
 // GpuPathTracer
 //
 //-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif
#define USE_DEBUG_EXCEPTIONS true

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "GpuPathTracer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

optix::Context        context;
uint32_t       width = 1024u;
uint32_t       height = 768u;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;
optix::Aabb    aabb;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Pathtracing
int max_depth = 10;
int frame_number = 1;
bool recount = false;

//postprocessing
Buffer denoised_buffer;
Buffer empty_buffer;
Buffer training_data_buffer;

PostprocessingStage tonemap_stage;
PostprocessingStage denoise_stage;
CommandList commandlist;
bool initDenoiser = true;
float denoise_blend = 0.0f;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
Buffer getAlbedoBuffer();
Buffer getNormalBuffer();
void destroyContext();
void registerExitHandler();
void createContext(int usage_report_level, UsageReportLogger* logger);
void loadMeshes(std::vector<std::string> filenames, std::vector<float3> positions, std::vector<int> mesh_brdf_types);
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context["output_buffer"]->getBuffer();
}
Buffer getTonemappedBuffer()
{
    return context["tonemapped_buffer"]->getBuffer();
}
Buffer getAlbedoBuffer()
{
    return context["input_albedo_buffer"]->getBuffer();
}
Buffer getNormalBuffer()
{
    return context["input_normal_buffer"]->getBuffer();
}


void destroyContext()
{
    if (context)
    {
        context->destroy();
        context = 0;
    }
}


struct UsageReportLogger
{
    void log(int lvl, const char* tag, const char* msg)
    {
        std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
    }
};

// Static callback
void usageReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
    // Route messages to a C++ object (the "logger"), as a real app might do.
    // We could have printed them directly in this simple case.

    UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>(cbdata);
    logger->log(lvl, tag, msg);
}

void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
    atexit(destroyContext);
#endif
}


void createContext(int usage_report_level, UsageReportLogger* logger)
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount(2);
    context->setStackSize(9280);
    context->setEntryPointCount(1);
    context->setMaxTraceDepth(max_depth+2);
    context->setMaxCallableProgramDepth(10);
    if (usage_report_level > 0)
    {
        context->setUsageReportCallback(usageReportCallback, usage_report_level, logger);
    }

    context["scene_epsilon"]->setFloat(1.e-4f);
    context["max_depth"]->setUint(max_depth);

    Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(buffer);
    denoised_buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["denoised_buffer"]->set(denoised_buffer);

    //TextureSampler tex_sampler = context->createTextureSampler();
    //tex_sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    //tex_sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    //tex_sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    //tex_sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    //tex_sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    //tex_sampler->setMaxAnisotropy(1.0f);
    //tex_sampler->setBuffer(0, 0, tex_buffer);
    //load box texture
    //const float3 box_default_color = make_float3(1.0f, 0.0f, 0.0f);
    //const std::string box_texpath = std::string(sutil::samplesDir()) + "/scenes/wood-chest-photoscan-pbr/source/chest_basecolor.jpg";
    //context["box_texture"]->setTextureSampler(sutil::loadTexture(context, box_texpath, box_default_color));

    // Accumulation buffer
    Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, width, height);
    context["accum_buffer"]->set(accum_buffer);

#if DENOISER_TYPE == 2
    //empty_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0); 
    //training_data_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);
    Buffer tonemapped_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["tonemapped_buffer"]->set(tonemapped_buffer);
    Buffer albedo_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_albedo_buffer"]->set(albedo_buffer);
    Buffer normal_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_normal_buffer"]->set(normal_buffer);
#endif

    // Ray generation program
    const char* ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
    Program ray_gen_program = context->createProgramFromPTXString(ptx, "pinhole_camera");
    context->setRayGenerationProgram(0, ray_gen_program);

    // Exception program
    Program exception_program = context->createProgramFromPTXString(ptx, "exception");
    context->setExceptionProgram(0, exception_program);
    context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

    // Miss program
    const float3 default_color = make_float3(1000.0f, 0.0f, 0.0f);
    const std::string texpath = std::string(sutil::samplesDir()) + "/scenes/envmaps/001.hdr";
    context["envmap"]->setTextureSampler(sutil::loadTexture(context, texpath, default_color));
    context->setMissProgram(0, context->createProgramFromPTXString(ptx, "envmap_miss"));

#if USE_DEBUG_EXCEPTIONS
    context->setPrintEnabled(true);
    context->setPrintBufferSize(4096);
    context->setPrintLaunchIndex(width, height);
    context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif
}

void loadMeshes(std::vector<std::string> filenames, std::vector<float3> positions, std::vector<int> brdf_types)
{
    GeometryGroup geometry_group;
    geometry_group = context->createGeometryGroup();
    geometry_group->setAcceleration(context->createAcceleration("Trbvh"));

    // Setup closest and any hit programs for our meshes
    const char* ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
    Program any_hit = context->createProgramFromPTXString(ptx, "shadow");
    Program closest_hit = context->createProgramFromPTXString(ptx, "closest_hit_li");
    Program closest_hit_cook = context->createProgramFromPTXString(ptx, "closest_hit_li_cook");
    Program closest_hit_ggx = context->createProgramFromPTXString(ptx, "closest_hit_li_ggx");
    Program closest_hit_beckmann = context->createProgramFromPTXString(ptx, "closest_hit_li_beckmann");
    Program closest_hit_other = context->createProgramFromPTXString(ptx, "closest_hit_other");

    for (int i = 0; i < filenames.size(); ++i)
    {
        OptiXMesh mesh;
        mesh.context = context;
        mesh.use_tri_api = use_tri_api;
        mesh.ignore_mats = ignore_mats;

        // Change default programs
        mesh.closest_hit = closest_hit;
        if(brdf_types[i] == 1)
            mesh.closest_hit = closest_hit_cook;
        else if (brdf_types[i] == 2)
            mesh.closest_hit = closest_hit_ggx;
        else if (brdf_types[i] == 3)
            mesh.closest_hit = closest_hit_beckmann;
        else if (brdf_types[i] == 9)
            mesh.closest_hit = closest_hit_other;
        mesh.any_hit = any_hit;
        //mesh.bounds = bounds;
        //mesh.intersection = intersection;

        // Optix loads our mesh
        loadMesh(filenames[i], mesh, Matrix4x4::translate(positions[i]));


        // Add to BVH
        aabb.include(mesh.bbox_min, mesh.bbox_max);
        geometry_group->addChild(mesh.geom_instance);
    }

    context["top_object"]->set(geometry_group);
    context["top_shadower"]->set(geometry_group);

}


void setupCamera()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    //camera_eye = aabb.center() + make_float3(-40.0f, 25.0f, max_dim * 1.5f);
    camera_eye = make_float3(-60.0f, 80.0f, 60.0f);
    camera_lookat = aabb.center();
    camera_up = make_float3(0.0f, 1.0f, 0.0f);

    camera_rotate = Matrix4x4::identity();
}


void setupLights()
{
    const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

    BasicLight lights[] = {
        { make_float3(10.0f,  40.0f ,  10.0f), make_float3(1.0f, 1.0f, 1.0f), 1, 2500.0f }
    };
    //lights[0].pos *= max_dim;
    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();
    context["lights"]->set(light_buffer);

    ParallelogramLight plight;
    plight.corner = make_float3(40, 40, 30);
    plight.v1 = make_float3(-30.0f, 0.0f, 0.0f);
    plight.v2 = make_float3(0.0f, 0.0f, 20.0f);
    plight.normal = normalize(cross(plight.v1, plight.v2));
    plight.emission = make_float3(15.0f, 15.0f, 5.0f);

    Buffer plight_buffer = context->createBuffer(RT_BUFFER_INPUT);
    plight_buffer->setFormat(RT_FORMAT_USER);
    plight_buffer->setElementSize(sizeof(ParallelogramLight));
    plight_buffer->setSize(1u);
    memcpy(plight_buffer->map(), &plight, sizeof(plight));
    plight_buffer->unmap();
    context["plights"]->setBuffer(plight_buffer);

}


void updateCamera()
{
    const float vfov = 35.0f;
    const float aspect_ratio = static_cast<float>(width) /
        static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
        camera_u, camera_v, camera_w, true);

    const Matrix4x4 frame = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

    camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
    camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
    camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
        camera_u, camera_v, camera_w, true);

    camera_rotate = Matrix4x4::identity();

    if (recount)
    {
        frame_number = 1;
        recount = false;
    }

    context["frame_number"]->setUint(frame_number++);
    context["eye"]->setFloat(camera_eye);
    context["U"]->setFloat(camera_u);
    context["V"]->setFloat(camera_v);
    context["W"]->setFloat(camera_w);

    const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat).inverse();
    Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);
    context["normal_matrix"]->setMatrix3x3fv(false, normal_matrix.getData());
}


void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(SAMPLE_NAME);
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutDisplay);
    glutReshapeFunc(glutResize);
    glutKeyboardFunc(glutKeyboardPress);
    glutMouseFunc(glutMousePress);
    glutMotionFunc(glutMouseMotion);

    registerExitHandler();

    glutMainLoop();
}

void setupDenoiser()
{
    tonemap_stage = context->createBuiltinPostProcessingStage("TonemapperSimple");
    denoise_stage = context->createBuiltinPostProcessingStage("DLDenoiser");
    //if (training_data_buffer)
    //{
    //    Variable training_buff = denoise_stage->declareVariable("training_data_buffer");
    //    training_buff->set(training_data_buffer);
    //}
    tonemap_stage->declareVariable("input_buffer")->set(getOutputBuffer());
    tonemap_stage->declareVariable("output_buffer")->set(getTonemappedBuffer());
    tonemap_stage->declareVariable("exposure")->setFloat(1.0f);
    tonemap_stage->declareVariable("gamma")->setFloat(2.2f);
    tonemap_stage->declareVariable("hdr")->setFloat(1);
    denoise_stage->declareVariable("input_buffer")->set(getTonemappedBuffer());
    denoise_stage->declareVariable("output_buffer")->set(denoised_buffer);
    denoise_stage->declareVariable("hdr")->setUint(0);
    denoise_stage->declareVariable("blend")->setFloat(denoise_blend);
    denoise_stage->declareVariable("input_albedo_buffer");
    denoise_stage->declareVariable("input_normal_buffer");
    commandlist = context->createCommandList();
    commandlist->appendLaunch(0, width, height);
    commandlist->appendPostprocessingStage(tonemap_stage, width, height);
    commandlist->appendPostprocessingStage(denoise_stage, width, height);
    commandlist->finalize();

    initDenoiser = false;
}

//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();
    context->launch(0, width, height);

#if DENOISER_TYPE == 2
    if (initDenoiser)
    {
        setupDenoiser();
    }
    Variable(denoise_stage->queryVariable("blend"))->setFloat(denoise_blend);
    commandlist->execute();
    sutil::displayBufferGL(denoised_buffer);
#else
    sutil::displayBufferGL(getOutputBuffer());
#endif

    {
        static unsigned frame_count = 0;
        sutil::displayFps(frame_count++);
    }
#if DENOISER_TYPE == 3
    char str[64];
    sprintf(str, "Accumulating frames #%d", frame_number);
    sutil::displayText(str, 10, 55);
#endif

    glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{

    switch (k)
    {
    case('q'):
    case(27): // ESC
    {
        destroyContext();
        exit(0);
    }
    case('s'):
    {
        const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
        std::cerr << "Saving current frame to '" << outputImage << "'\n";
        sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());
        break;
    }
    }
}


void glutMousePress(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_button = button;
        mouse_prev_pos = make_int2(x, y);
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion(int x, int y)
{
    if (mouse_button == GLUT_RIGHT_BUTTON)
    {
        const float dx = static_cast<float>(x - mouse_prev_pos.x) /
            static_cast<float>(width);
        const float dy = static_cast<float>(y - mouse_prev_pos.y) /
            static_cast<float>(height);
        const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
        const float scale = fminf(dmax, 0.9f);
        camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
        recount = true;
    }
    else if (mouse_button == GLUT_LEFT_BUTTON)
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x / width, to.y / height };

        camera_rotate = arcball.rotate(b, a);
        recount = true;
    }

    mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
    if (w == (int)width && h == (int)height) return;
    recount = true;
    width = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer(getOutputBuffer(), width, height);
    sutil::resizeBuffer(getAlbedoBuffer(), width, height);
    sutil::resizeBuffer(getNormalBuffer(), width, height);

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0)
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "  -r | --report <LEVEL>     Enable usage reporting and report level [1-3].\n"
        "  -i | --ignore-materials   Ignore materials in the mesh file.\n"
        "       --no-triangle-api    Disable the Triangle API.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}

int main(int argc, char** argv)
{
    std::string out_file;
#if SCENES == 1 
    std::vector<std::string> mesh_filenames =
    {
        // Cook Torrance
        std::string(sutil::samplesDir()) + "/scenes/sphere.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere1.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere2.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere3.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere4.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere5.obj",
        // GGX
        std::string(sutil::samplesDir()) + "/scenes/sphere.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere1.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere2.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere3.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere4.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere5.obj",
        // Beckmann
        std::string(sutil::samplesDir()) + "/scenes/sphere.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere1.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere2.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere3.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere4.obj",
        std::string(sutil::samplesDir()) + "/scenes/sphere5.obj",
    };
    std::vector<float3> mesh_positions =
    {
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 5.0f),
        make_float3(0.0f, 0.0f, 10.0f),
        make_float3(0.0f, 0.0f, 15.0f),
        make_float3(0.0f, 0.0f, 20.0f),
        make_float3(0.0f, 0.0f, 25.0f),

        make_float3(5.0f, 0.0f, 0.0f),
        make_float3(5.0f, 0.0f, 5.0f),
        make_float3(5.0f, 0.0f, 10.0f),
        make_float3(5.0f, 0.0f, 15.0f),
        make_float3(5.0f, 0.0f, 20.0f),
        make_float3(5.0f, 0.0f, 25.0f),

        make_float3(10.0f, 0.0f, 0.0f),
        make_float3(10.0f, 0.0f, 5.0f),
        make_float3(10.0f, 0.0f, 10.0f),
        make_float3(10.0f, 0.0f, 15.0f),
        make_float3(10.0f, 0.0f, 20.0f),
        make_float3(10.0f, 0.0f, 25.0f),
    };
    std::vector<int> mesh_brdf_types =
    {
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
    };
#elif SCENES == 0
    std::vector<std::string> mesh_filenames =
    {
        std::string(sutil::samplesDir()) + "/scenes/NewShip.obj",
        std::string(sutil::samplesDir()) + "/scenes/LandingPad.obj",
    };
    std::vector<float3> mesh_positions =
    {
        make_float3(0.0f, 10.0f, 0.0f),
        make_float3(0.0f, 0.0f, 0.0f),
    };
    std::vector<int> mesh_brdf_types =
    {
        0, 0,
    };
#elif SCENES == 2
    std::vector<std::string> mesh_filenames =
    {
        std::string(sutil::samplesDir()) + "/scenes/fish/02_02_position2.obj",
    };
    std::vector<float3> mesh_positions =
    {
        make_float3(0.0f, 10.0f, 0.0f),
    };
    std::vector<int> mesh_brdf_types =
    {
        9,
    };
#endif
    int usage_report_level = 0;

    try
    {
        glutInitialize(&argc, argv);

#ifndef __APPLE__
        glewInit();
#endif

        UsageReportLogger logger;
        createContext(usage_report_level, &logger);
        loadMeshes(mesh_filenames, mesh_positions, mesh_brdf_types);
        setupCamera();
        setupLights();

        context->validate();
        if (out_file.empty())
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch(0, width, height);
            sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer());
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH(context->get())
}

