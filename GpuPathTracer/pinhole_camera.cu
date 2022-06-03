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

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optix_world.h>
#include <common.h>
#include "helpers.h"
#include "brdf_helper.cu"

using namespace optix;

struct PerRayData_radiance
{
  float3  result;
  float3  radiance;
  float3  importance;
  float3  origin;
  float3  direction;
  float3  albedo;
  float3  normal;
  int    depth;
  unsigned int seed;
  int  done;
};

struct PerRayData_shadow
{
    float3 visibility;
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(Matrix3x3, normal_matrix, , );//TODO:

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              input_albedo_buffer;
rtBuffer<float4, 2>              input_normal_buffer;
rtBuffer<float4, 2>              accum_buffer;

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

//#define TIME_VIEW
rtBuffer<BasicLight>        lights;
rtBuffer<ParallelogramLight>        plights;
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(unsigned int, max_depth, , );
rtDeclareVariable(unsigned int, frame_number, , );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

RT_PROGRAM void pinhole_camera()
{
  //TODO: might have issue
  const size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);
  //const float jitter_x = rnd(seed);
  //const float jitter_y = rnd(seed);
  //const float2 jitter = make_float2(jitter_x, jitter_y);
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  //const float2 d = (make_float2(launch_index) + jitter) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x * U + d.y * V + W);

#if DENOISER_TYPE == 0
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
  PerRayData_radiance prd;
  prd.importance = make_float3(1.0f);
  prd.depth = 0;
  prd.seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);
  rtTrace(top_object, ray, prd ) ;
  output_buffer[launch_index] = make_float4(prd.result);
  if (prd.result.x > 1.0f || prd.result.y > 1.0f || prd.result.z > 1.0f)
  {
      output_buffer[launch_index] = make_float4(1.0f);
  }
#elif DENOISER_TYPE == 1
  float3 result = make_float3(0.0f);
  int number_of_samples = NUMBER_OF_SAMPLES;
  for (int i = 0; i < number_of_samples; i++)
  {
      optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
      PerRayData_radiance prd;
      prd.importance = make_float3(1.0f);
      prd.depth = 0;
      prd.seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1 + i);
      rtTrace(top_object, ray, prd);
      result = result + prd.result;
  }
  result = result / number_of_samples;
  output_buffer[launch_index] = make_float4(result);
#elif DENOISER_TYPE == 2
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
  PerRayData_radiance prd;
  prd.importance = make_float3(1.0f);
  prd.depth = 0;
  prd.seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);
  rtTrace(top_object, ray, prd);

  float3 result = prd.result;
  float3 albedo = prd.albedo;
  float3 normal = (length(prd.normal) > 0.f) ? normalize(normal_matrix * prd.normal) : make_float3(0., 0., 1.);

  if (frame_number > 1)
  {
      float a = 1.0f / (float)frame_number;
      float3 old_result = make_float3(output_buffer[launch_index]);
      float3 old_albedo = make_float3(input_albedo_buffer[launch_index]);
      float3 old_normal = make_float3(input_normal_buffer[launch_index]);
      output_buffer[launch_index] = make_float4(lerp(old_result, result, a));
      input_albedo_buffer[launch_index] = make_float4(lerp(old_albedo, albedo, a), 1.0f);

      float3 accum_normal = lerp(old_normal, normal, a);
      input_normal_buffer[launch_index] = make_float4((length(accum_normal) > 0.f) ? normalize(accum_normal) : normal, 1.0f);
  }
  else
  {
      output_buffer[launch_index] = make_float4(result);
      input_albedo_buffer[launch_index] = make_float4(albedo, 1.0f);
      input_normal_buffer[launch_index] = make_float4(normal, 1.0f);
  }
  output_buffer[launch_index] = make_float4(prd.result);
#elif DENOISER_TYPE == 3
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
  PerRayData_radiance prd;
  prd.importance = make_float3(1.0f);
  prd.depth = 0;
  prd.seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);
  rtTrace(top_object, ray, prd);
  float4 accum_result = make_float4(0.0f);
  if (frame_number == 1) {
      accum_result = make_float4(prd.result);
  }
  else {
      accum_result = accum_buffer[launch_index] + make_float4(prd.result);
  }
  accum_buffer[launch_index] = accum_result;
  output_buffer[launch_index] = accum_result / frame_number;
#endif
}

//RT_PROGRAM void pinhole_camera()
//{
//    int rr_begin_depth = 1;
//    unsigned int sqrt_num_samples = 2;
//
//    size_t2 screen = output_buffer.size();
//
//    float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
//    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;
//
//    float2 jitter_scale = inv_screen / sqrt_num_samples;
//    unsigned int samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
//    float3 result = make_float3(0.0f);
//
//    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);
//    do
//    {
//        // Sample pixel using jittering
//        unsigned int x = samples_per_pixel % sqrt_num_samples;
//        unsigned int y = samples_per_pixel / sqrt_num_samples;
//        float2 jitter = make_float2(x - rnd(seed), y - rnd(seed));
//        float2 d = pixel + jitter * jitter_scale;
//        float3 ray_origin = eye;
//        float3 ray_direction = normalize(d.x * U + d.y * V + W);
//
//        // Initialze per-ray data
//        PerRayData_radiance prd;
//        prd.result = make_float3(0.f);
//        prd.importance = make_float3(1.f);
//        prd.seed = seed;
//        prd.depth = 0;
//        prd.done = false;
//
//        for (int i = 0; i < max_depth; i++)
//        //for (; ; )
//        {
//            Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
//            rtTrace(top_object, ray, prd);
//
//            if (prd.done)
//            {
//                // We have hit the background
//                prd.result += prd.radiance * prd.importance;
//                break;
//            }
//
//            // Russian roulette termination 
//            if (prd.depth >= rr_begin_depth)
//            {
//                float pcont = fmaxf(prd.importance);
//                if (rnd(prd.seed) >= pcont)
//                    break;
//                prd.importance /= pcont;
//            }
//
//            prd.depth++;
//            prd.result += prd.radiance * prd.importance;
//
//            // Update ray data for the next path segment
//            ray_origin = prd.origin;
//            ray_direction = prd.direction;
//        }
//
//        result += prd.result;
//        seed = prd.seed;
//    } while (--samples_per_pixel);
//
//    // Update the output buffer
//    float3 pixel_color = result / (sqrt_num_samples * sqrt_num_samples);
//
//    if (frame_number > 1)
//    {
//        float a = 1.0f / (float)frame_number;
//        float3 old_color = make_float3(output_buffer[launch_index]);
//        output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
//    }
//    else
//    {
//        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
//    }
//}


RT_PROGRAM void exception()
{
  rtPrintExceptionDetails();
  output_buffer[launch_index] = make_float4(bad_color);
}

rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
    prd_radiance.result = make_float3(tex2D(envmap, u, v)) * prd_radiance.importance;
    prd_radiance.done = true;
#if DENOISER_TYPE == 2
    if (prd_radiance.depth == 0)
    {
        prd_radiance.albedo = make_float3(0.0f);
        prd_radiance.normal = make_float3(0.0f);
    }
#endif
}

RT_PROGRAM void closest_hit_li()
{
    float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
    float3 color = make_float3(0.0f);
    float3 hit_point = ray.origin + t_hit * ray.direction;

    float Ldist = 0.f;
    float3 wi = make_float3(0.0f);
    float3 L = make_float3(0.0f);
#if POINT_LIGHT
    BasicLight light = lights[0];
    float3 L = normalize(light.pos - hit_point);
    //shadow ray
    PerRayData_shadow shadow_prd;
    shadow_prd.visibility = make_float3(1.0f);
    Ldist = length(light.pos - hit_point);
    optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
    rtTrace(top_shadower, shadow_ray, shadow_prd);
    float3 light_visibility = shadow_prd.visibility;
    wi = normalize(light.pos - hit_point);

    //direct lighting
    if (fmaxf(light_visibility) > 0.0f) {
        float falloff_factor = 1.0f / (Ldist * Ldist);
        float3 Li = light.intensity_multiplier * falloff_factor * light.color;
        color += prd_radiance.importance * Li * linearblend_reflectivity_f(wi, -ray.direction, ffnormal) * max(0.0f, dot(wi, ffnormal));
    }
#endif
#if AREA_LIGHT
    // Choose random point on ParallelogramLight
    ParallelogramLight plight = plights[0];
    float z1 = rnd(prd_radiance.seed);
    float z2 = rnd(prd_radiance.seed);
    float3 light_pos = plight.corner + plight.v1 * z1 + plight.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    Ldist = length(light_pos - hit_point);
    L = normalize(light_pos - hit_point);
    float  nDl = dot(ffnormal, L);
    float  LnDl = dot(plight.normal, L);
    wi = normalize(light_pos - hit_point);

    //shadow ray
    if (nDl > 0.0f && LnDl > 0.0f)
    {
        PerRayData_shadow shadow_prd;
        shadow_prd.visibility = make_float3(1.0f);
        optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        float3 light_visibility = shadow_prd.visibility;

        if (fmaxf(light_visibility) > 0.0f)
        {
            const float A = length(cross(plight.v1, plight.v2));
            const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
            color += plight.emission * weight * linearblend_reflectivity_f(wi, -ray.direction, ffnormal);
        }
    }
#endif

    //emissive lighting
    color += prd_radiance.importance * Ke * Kd;

    //indirect lighting
    float pdf = 1.0f;
    float3 brdf = linearblend_samplewi(prd_radiance.seed, wi, -ray.direction, ffnormal, pdf);
    float cosine_term = abs(dot(wi, ffnormal));
    if (pdf < scene_epsilon){
        prd_radiance.result = color;
        return;
    }
    float3 importance = prd_radiance.importance * (brdf * cosine_term) / pdf;
    if (importance.x == 0 &&
        importance.y == 0 &&
        importance.z == 0){
        prd_radiance.result = color;
        return;
    }
#if DENOISER_TYPE == 2
    if (prd_radiance.depth == 0)
    {
        prd_radiance.albedo = Kd;
        prd_radiance.normal = ffnormal;
    }
#endif   
    // Russian roulette termination 
    if (prd_radiance.depth >= 1)
    {
        float pcont = fmaxf(prd_radiance.importance);
        if (rnd(prd_radiance.seed) >= pcont)
        {
            prd_radiance.result = color;
            return;
        }
    }
    if (prd_radiance.depth < max_depth) {
        //float3 R = reflect(ray.direction, ffnormal);
        Ray reflection_ray = make_Ray(hit_point - scene_epsilon * ray.direction, wi, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);

        PerRayData_radiance reflection_prd;
        reflection_prd.importance = importance;
        reflection_prd.result = color;
        reflection_prd.depth = prd_radiance.depth + 1;
        reflection_prd.seed = prd_radiance.seed;
        rtTrace(top_object, reflection_ray, reflection_prd);
        color += reflection_prd.result;
    }

    prd_radiance.result = color;
}

RT_PROGRAM void closest_hit_li_cook()
{
    float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
    float3 color = make_float3(0.0f);
    float3 hit_point = ray.origin + t_hit * ray.direction;

    for (int i = 0; i < lights.size(); ++i) {
        BasicLight light = lights[i];
        float3 L = normalize(light.pos - hit_point);

        //shadow ray
        PerRayData_shadow shadow_prd;
        shadow_prd.visibility = make_float3(1.0f);
        float Ldist = length(light.pos - hit_point);
        optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        float3 light_visibility = shadow_prd.visibility;
        float3 wi = normalize(light.pos - hit_point);

        //direct lighting
        if (fmaxf(light_visibility) > 0.0f) {
            float falloff_factor = 1.0f / (Ldist * Ldist);
            float3 Li = light.intensity_multiplier * falloff_factor * light.color;
            color += prd_radiance.importance * Li * linearblend_reflectivity_f(wi, -ray.direction, ffnormal, 1.f) * max(0.0f, dot(wi, ffnormal));
        }

        //emissive lighting
        color += prd_radiance.importance * Ke * Kd;
    }

    prd_radiance.result = color;
}

RT_PROGRAM void closest_hit_li_ggx()
{
    float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
    float3 color = make_float3(0.0f);
    float3 hit_point = ray.origin + t_hit * ray.direction;

    for (int i = 0; i < lights.size(); ++i) {
        BasicLight light = lights[i];
        float3 L = normalize(light.pos - hit_point);

        //shadow ray
        PerRayData_shadow shadow_prd;
        shadow_prd.visibility = make_float3(1.0f);
        float Ldist = length(light.pos - hit_point);
        optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        float3 light_visibility = shadow_prd.visibility;
        float3 wi = normalize(light.pos - hit_point);

        //direct lighting
        if (fmaxf(light_visibility) > 0.0f) {
            float falloff_factor = 1.0f / (Ldist * Ldist);
            float3 Li = light.intensity_multiplier * falloff_factor * light.color;
            color += prd_radiance.importance * Li * linearblend_reflectivity_f(wi, -ray.direction, ffnormal, 2.f) * max(0.0f, dot(wi, ffnormal));
        }

        //emissive lighting
        color += prd_radiance.importance * Ke * Kd;
    }

    prd_radiance.result = color;
}

RT_PROGRAM void closest_hit_li_beckmann()
{
    float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
    float3 color = make_float3(0.0f);
    float3 hit_point = ray.origin + t_hit * ray.direction;

    for (int i = 0; i < lights.size(); ++i) {
        BasicLight light = lights[i];
        float3 L = normalize(light.pos - hit_point);

        //shadow ray
        PerRayData_shadow shadow_prd;
        shadow_prd.visibility = make_float3(1.0f);
        float Ldist = length(light.pos - hit_point);
        optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        float3 light_visibility = shadow_prd.visibility;
        float3 wi = normalize(light.pos - hit_point);

        //direct lighting
        if (fmaxf(light_visibility) > 0.0f) {
            float falloff_factor = 1.0f / (Ldist * Ldist);
            float3 Li = light.intensity_multiplier * falloff_factor * light.color;
            color += prd_radiance.importance * Li * linearblend_reflectivity_f(wi, -ray.direction, ffnormal, 3.f) * max(0.0f, dot(wi, ffnormal));
        }

        //emissive lighting
        color += prd_radiance.importance * Ke * Kd;
    }

    prd_radiance.result = color;
}

//rtTextureSampler<float4, 2> box_texture;
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
RT_PROGRAM void closest_hit_other()
{
    return;
}

RT_PROGRAM void shadow()
{
    prd_shadow.visibility = make_float3(0.0f);
	rtTerminateRay();
}