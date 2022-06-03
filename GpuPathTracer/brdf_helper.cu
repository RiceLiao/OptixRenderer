#include <optixu/optixu_math_namespace.h>
#include <optix.h>
#include <optix_math.h>

#include "common.h"
#include "random.h"
#include "helpers.h"


//////////////////////////////// //////////////////////////////// 
//
//						Material parameters
//
//////////////////////////////// //////////////////////////////// 

rtDeclareVariable(float3, Kd, , );          // Diffuse
rtDeclareVariable(float3, Ks, , );			// Specular
rtDeclareVariable(float3, Kr, , );			// Reflective
rtDeclareVariable(float3, Ka, , );			// Ambient
rtDeclareVariable(float, phong_exp, , );	// phong_exp
rtDeclareVariable(float, Pm, , );			// Metalness
rtDeclareVariable(float, Pr, , );			// Shininess / Roughness
rtDeclareVariable(float, Ps, , );			// Fresnel
rtDeclareVariable(float3, Tf, , );			// Transparency
rtDeclareVariable(float3, Ke, , );			// Emissive
rtDeclareVariable(int, Kd_mapped, , );	    // Has textures



//////////////////////////////// //////////////////////////////// 
//
//						BRDF Evaluation
// 
//////////////////////////////// //////////////////////////////// 
RT_CALLABLE_PROGRAM float3 perpendicular(const float3& v)
{
	if (fabsf(v.x) < fabsf(v.y))
	{
		return make_float3(0.0f, -v.z, v.y);
	}
	return make_float3(-v.z, 0.0f, v.x);
}

RT_CALLABLE_PROGRAM float3 new_cosine_sample_hemisphere(unsigned int seed)
{
	float phi, r;
	float u1 = rnd(seed);
	float u2 = rnd(seed);
    const float a = 2.0f * u1 - 1.0f;
    const float b = 2.0f * u2 - 1.0f;

    if (a == 0.0f && b == 0.0f)
    {
        return make_float3(0.0f, 0.0f, 1.0f);
    }

    if (a >= -b)
    {
        if (a > b)
        {
            r = a;
            if (b > 0.0f)
                phi = b / r;
            else
                phi = 8.0f + b / r;
        }
        else
        {
            r = b;
            phi = 2.0f - (a / r);
        }
    }
    else
    {
        if (a <= b)
        {
            r = -a;
            phi = 4.0f - b / r;
        }
        else
        {
            r = -b;
            phi = 6.0f + a / r;
        }
    }
    phi *= float(M_PIf) / 4.0f;
	float sample_x = r * cosf(phi);
	float sample_y = r * sinf(phi);
	float z = sqrtf(fmaxf(0.0f, 1.0f - sample_x * sample_x - sample_y * sample_y));
    return make_float3(sample_x, sample_y, z);
}

RT_CALLABLE_PROGRAM float blinn_d(const float& s, const float& n_dot_h)
{
	return (s + 2) / (2 * M_PIf) * powf(n_dot_h, s);
}

RT_CALLABLE_PROGRAM float beckmann_d(const float& s, const float& n_dot_h)
{
	float n_dot_h_2 = n_dot_h * n_dot_h;
	return exp((n_dot_h_2 - 1) / (s * s * n_dot_h_2)) / (M_PIf * s * s * n_dot_h_2 * n_dot_h_2);
}

RT_CALLABLE_PROGRAM float ggx_d(const float& s, const float& n_dot_h)
{
	float a2 = s * s;
	float d = ((n_dot_h * a2 - n_dot_h) * n_dot_h + 1);
	return a2 / (d * d * M_PIf);
}

RT_CALLABLE_PROGRAM float blinn_g(const float& n_dot_h, const float& n_dot_o, const float& n_dot_i, const float& o_dot_h)
{
	float m1 = 2 * n_dot_h * n_dot_o / o_dot_h;
	float m2 = 2 * n_dot_h * n_dot_i / o_dot_h;
	return min(1.0f, min(m1, m2));
}

RT_CALLABLE_PROGRAM float smith_g(const float& n_dot_o, const float& n_dot_i, const float& s)
{
	float k = s * s / 2;
	float g_o = n_dot_o / (n_dot_o * (1 - k) + k);
	float g_i = n_dot_i / (n_dot_i * (1 - k) + k);
	return g_o * g_i;
}

RT_CALLABLE_PROGRAM float3 diffuse_f(float3 wi, float3 wo, float3 n)
{

	if (dot(wi, n) <= 0.0f)
		return make_float3(0.0f);
	else if (!(signbit(dot(wi, n)) == signbit(dot(wo, n))))
		return make_float3(0.0f);
	else
		return (1.0f / M_PIf) * Kd;

}

RT_CALLABLE_PROGRAM float3 blinnphong_f(float3 wi, float3 wo, float3 n, float brdf_type = 1.f)
{
	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	//reflection
	if (dot(n, wi) <= 0.0f || dot(n, wo) <= 0.0f)
		blinnphong_reflection_brdf = make_float3(0.0f);

	float3 wh = normalize(wi + wo);
	float n_dot_h = dot(n, wh);
	float n_dot_o = dot(n, wo);
	float n_dot_i = dot(n, wi);
	float o_dot_h = dot(wo, wh);
	float F = Ps + ((1.0f - Ps) * powf(1.0f - abs(dot(wh, wi)), 5.0f));
	float s = Pr;
	float D = 0.0f;
	float G = 0.0f;
	if (brdf_type == 1.f) { // Cook-Torrance
		D = blinn_d(s, n_dot_h);
		G = blinn_g(n_dot_h, n_dot_o, n_dot_i, o_dot_h);
	} else if (brdf_type == 2.f) {// GGX
		D = ggx_d(s, n_dot_h);
		G = smith_g(n_dot_o, n_dot_i, s);
	} else if (brdf_type == 3.f) {// Beckmann
		D = beckmann_d(s, n_dot_h);
		G = smith_g(n_dot_o, n_dot_i, s);
	}
	if (brdf_type == 2.f) {
		blinnphong_reflection_brdf = make_float3(F * D * G / (4 * n_dot_o));
	}
	else{
		blinnphong_reflection_brdf = make_float3(F * D * G / (4 * n_dot_o * n_dot_i));
	}
		
	blinnphong_refraction_brdf = make_float3(1.0f - F) * diffuse_f(wi, wo, n);
	blinnphong_brdf = blinnphong_reflection_brdf + blinnphong_refraction_brdf;
	return blinnphong_brdf;
}

RT_CALLABLE_PROGRAM float3 blinnphongmetal_f(float3 wi, float3 wo, float3 n, float brdf_type = 1.f)
{
	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	//reflection
	if (dot(n, wi) <= 0.0f || dot(n, wo) <= 0.0f)
		blinnphong_reflection_brdf = make_float3(0.0f);

	float3 wh = normalize(wi + wo);
	float n_dot_h = dot(n, wh);
	float n_dot_o = dot(n, wo);
	float n_dot_i = dot(n, wi);
	float o_dot_h = dot(wo, wh);
	float F = Ps + ((1.0f - Ps) * powf(1.0f - abs(dot(wh, wi)), 5.0f));
	float s = Pr;
	float D = 0.0f;
	float G = 0.0f;
	if (brdf_type == 1.f) { // Cook-Torrance
		D = blinn_d(s, n_dot_h);
		G = blinn_g(n_dot_h, n_dot_o, n_dot_i, o_dot_h);
	}
	else if (brdf_type == 2.f) {// GGX
		D = ggx_d(s, n_dot_h);
		G = smith_g(n_dot_o, n_dot_i, s);
	}
	else if (brdf_type == 3.f) {// Beckmann
		D = beckmann_d(s, n_dot_h);
		G = smith_g(n_dot_o, n_dot_i, s);
	}
	if (brdf_type == 2.f) {
		blinnphong_reflection_brdf = F * D * G / (4 * n_dot_o) * Kd;
	}
	else {
		blinnphong_reflection_brdf = F * D * G / (4 * n_dot_o * n_dot_i) * Kd;
	}
	blinnphong_reflection_brdf = F * D * G / (4 * n_dot_o * n_dot_i) * Kd;
	blinnphong_refraction_brdf = make_float3(0.0f);
	blinnphong_brdf = blinnphong_reflection_brdf + blinnphong_refraction_brdf;
	return blinnphong_brdf;
}

// linear blend brdf between reflective and metalness blinn phong
RT_CALLABLE_PROGRAM float3 linearblend_metal_f(float3 wi, float3 wo, float3 n, float brdf_type = 1.f)
{
	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n, brdf_type) + (1.0f -Pm) * blinnphong_f(wi, wo, n, brdf_type);
	return metal_blend;
}

RT_CALLABLE_PROGRAM float3 linearblend_reflectivity_f(float3 wi, float3 wo, float3 n, float brdf_type = 1.f)
{
	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n, brdf_type) + (1.0f - Pm) * blinnphong_f(wi, wo, n, brdf_type);
	float3 reflectivity_blend = Ks * metal_blend + (1.0f - Ks) * diffuse_f(wi, wo, n);
	return reflectivity_blend;
}


//////////////////////////////// //////////////////////////////// 
//
//						Importance sampling
// 
//////////////////////////////// //////////////////////////////// 


RT_CALLABLE_PROGRAM float3 diffuse_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float3 tangent = normalize(perpendicular(n));
	float3 bitangent = normalize(cross(tangent, n));

	float z1 = rnd(seed);
	float z2 = rnd(seed);
	float3 sample;
	optix::cosine_sample_hemisphere(z1, z2, sample);
	
	//float2 sample2d;
	//optix::square_to_disk(sample2d);
	//float z = sqrtf(fmaxf(0.0f, 1.0f - sample2d.x * sample2d.x - sample2d.y * sample2d.y));
	//float3 sample = new_cosine_sample_hemisphere(seed);
	

	wi = normalize(sample.x * tangent + sample.y * bitangent + sample.z * n);
	if (dot(wi, n) <= 0.0f)
		p = 0.0f;
	else
		p = max(0.0f, dot(n, wi)) / M_PIf;
	return diffuse_f(wi, wo, n);
}

RT_CALLABLE_PROGRAM float3 blinnphong_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{		

	float3 tangent = normalize(perpendicular(n));;
	float3 bitangent = normalize(cross(tangent, n));

	float phi = 2.0f * M_PIf * rnd(seed);
	float cos_theta = pow(rnd(seed), 1.0f / (Pr + 1));
	float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
	float3 wh = normalize(sin_theta * cos(phi) * tangent + sin_theta * sin(phi) * bitangent + cos_theta * n);

	if (dot(wo, n) <= 0.0f)
		return make_float3(0.0f);

	wi = normalize(2 * dot(wh, wo) * wh - wo);
	float pwh = (Pr + 1) * pow(dot(n, wh), Pr) / (2 * M_PIf);
	p = pwh / (4 * dot(wo, wh));
	p = p * 0.5f;

	if (rnd(seed) < 0.5f)
	{
		return blinnphong_f(wi, wo, n);
	}
	else
	{
		float3 brdf = diffuse_samplewi(seed, wi, wo, n, p);
		float f = Ps + (1.0f - Ps) * pow(1.0f - abs(dot(wh, wi)), 5.0f);
		return (1 - f) * brdf;
	}
}

RT_CALLABLE_PROGRAM float3 blinnphongmetal_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	return blinnphong_samplewi(seed, wi, wo, n, p);
}

RT_CALLABLE_PROGRAM float3 linearblend_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	p = 0.0f;

	// Reflectivity
	if (rnd(seed) < (Ks.x + Ks.y + Ks.z) / 3)
	{
		// Metalness
		if (rnd(seed) < Pm)
		{
			return blinnphongmetal_samplewi(seed, wi, wo, n, p);
		}
		else
		{
			return blinnphong_samplewi(seed, wi, wo, n, p);
		}
	}
	else
	{
		return diffuse_samplewi(seed, wi, wo, n, p);
	}
}