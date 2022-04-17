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

//TODO: little difference with the CPU version ( ks, ka )
rtDeclareVariable(float3, Kd, , );          // Diffuse
rtDeclareVariable(float3, Ks, , );			// Specular
rtDeclareVariable(float3, Kr, , );			// Reflective
rtDeclareVariable(float3, Ka, , );			// Ambient
rtDeclareVariable(float, phong_exp, , );	// phong_exp
rtDeclareVariable(float, Pm, , );			// Metalness
rtDeclareVariable(float, Pr, , );			// Shininess
rtDeclareVariable(float, Ps, , );			// Fresnel
rtDeclareVariable(float3, Tf, , );			// Transparency
rtDeclareVariable(float3, Ke, , );			// Emissive
rtDeclareVariable(int, Kd_mapped, , );	    // Has textures



//////////////////////////////// //////////////////////////////// 
//
//						BRDF Evaluation
// 
//////////////////////////////// //////////////////////////////// 

RT_CALLABLE_PROGRAM float3 diffuse_f(float3 wi, float3 wo, float3 n)
{

	if (dot(wi, n) <= 0.0f)
		return make_float3(0.0f);
	else if (!(signbit(dot(wi, n)) == signbit(dot(wo, n))))
		return make_float3(0.0f);
	else
		return (1.0f / M_PIf) * Kd;

}

RT_CALLABLE_PROGRAM float3 blinnphong_f(float3 wi, float3 wo, float3 n)
{
	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	//reflection
	if (dot(n, wi) <= 0.0f || dot(n, wo) <= 0.0f)
		blinnphong_reflection_brdf = make_float3(0.0f);

	float3 wh = normalize(wi + wo);
	float F = Ps + ((1.0f - Ps) * powf(1.0f - abs(dot(wh, wi)), 5.0f));
	float s = Pr;
	float D = (s + 2) / (2 * M_PI) * powf(dot(n, wh), s);
	float m1 = 2 * dot(n, wh) * dot(n, wo) / dot(wo, wh);
	float m2 = 2 * dot(n, wh) * dot(n, wi) / dot(wo, wh);
	float G = min(1.0f, min(m1, m2));
	blinnphong_reflection_brdf = make_float3(F * D * G / (4 * dot(n, wo) * dot(n, wi)));
	blinnphong_refraction_brdf = make_float3(1.0f - F) * diffuse_f(wi, wo, n); // TODO: little difference
	blinnphong_brdf = blinnphong_reflection_brdf + blinnphong_refraction_brdf;
	return blinnphong_brdf;
}

RT_CALLABLE_PROGRAM float3 blinnphongmetal_f(float3 wi, float3 wo, float3 n)
{
	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	//reflection
	if (dot(n, wi) <= 0.0f || dot(n, wo) <= 0.0f)
		blinnphong_reflection_brdf = make_float3(0.0f);

	float3 wh = normalize(wi + wo);
	float F = Ps + ((1.0f - Ps) * powf(1.0f - abs(dot(wh, wi)), 5.0f));
	float s = Pr;
	float D = (s + 2) / (2 * M_PI) * powf(dot(n, wh), s);
	float m1 = 2 * dot(n, wh) * dot(n, wo) / dot(wo, wh);
	float m2 = 2 * dot(n, wh) * dot(n, wi) / dot(wo, wh);
	float G = min(1.0f, min(m1, m2));
	blinnphong_reflection_brdf = F * D * G / (4 * dot(n, wo) * dot(n, wi)) * Kd;
	blinnphong_refraction_brdf = make_float3(0.0f);
	blinnphong_brdf = blinnphong_reflection_brdf + blinnphong_refraction_brdf;
	return blinnphong_brdf;
}

//RT_CALLABLE_PROGRAM float3 glass_f(float3 wi, float3 wo, float3 n)
//{
//	float3 glass_reflection_brdf;
//	float3 glass_refraction_brdf;
//	float3 glass_brdf;
//
//	//reflection
//	if (dot(n, wi) <= 0.0f || dot(n, wo) <= 0.0f)
//		glass_reflection_brdf = make_float3(0.0f);
//
//	float3 wh = normalize(wi + wo);
//	float F = Ps + ((1.0f - Ps) * powf(1.0f - abs(dot(wh, wi)), 5.0f));
//	float s = Pr;
//	float D = (s + 2) / (2 * M_PI) * powf(dot(n, wh), s);
//	float m1 = 2 * dot(n, wh) * dot(n, wo) / dot(wo, wh);
//	float m2 = 2 * dot(n, wh) * dot(n, wi) / dot(wo, wh);
//	float G = min(1.0f, min(m1, m2));
//	glass_reflection_brdf = F * D * G / (4 * dot(n, wo) * dot(n, wi)) * Kd;
//	glass_refraction_brdf = make_float3(0.0f);
//	glass_brdf = glass_reflection_brdf + glass_refraction_brdf;
//	return glass_brdf;
//}

// linear blend brdf between reflective and metalness blinn phong
RT_CALLABLE_PROGRAM float3 linearblend_metal_f(float3 wi, float3 wo, float3 n)
{
	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n) + (1.0f -Pm) * blinnphong_f(wi, wo, n);
	return metal_blend;
}

RT_CALLABLE_PROGRAM float3 linearblend_reflectivity_f(float3 wi, float3 wo, float3 n)
{
	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n) + (1.0f - Pm) * blinnphong_f(wi, wo, n);
	//float3 reflectivity_blend = Kr * metal_blend + (1.0f - Kr) * diffuse_f(wi, wo, n);
	float3 reflectivity_blend = Ks * metal_blend + (1.0f - Ks) * diffuse_f(wi, wo, n); //TODO
	return reflectivity_blend;
}

//RT_CALLABLE_PROGRAM float3 linearblend_glass_f(float3 wi, float3 wo, float3 n)
//{
//	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n) + (1.0f - Pm) * blinnphong_f(wi, wo, n);
//	float3 reflectivity_blend = Ks * metal_blend + (1.0f - Ks) * diffuse_f(wi, wo, n);
//	float3 glass_blend = Tf * reflectivity_blend + (1.0f - Tf) * glass_f(wi, wo, n);
//	return glass_blend;
//}


//////////////////////////////// //////////////////////////////// 
//
//						Importance sampling
// 
//////////////////////////////// //////////////////////////////// 


// Samples a new wi direction for pathtracing and returns a simple diffuse brdf
RT_CALLABLE_PROGRAM float3 diffuse_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float3 tangent;
	float3 bitangent;

	create_onb(n, tangent, bitangent);

	float z1 = rnd(seed);
	float z2 = rnd(seed);
	float3 sample;
	optix::cosine_sample_hemisphere(z1, z2, sample);

	wi = normalize(sample.x * tangent + sample.y * bitangent + sample.z * n);
	if (dot(wi, n) <= 0.0f)
		p = 0.0f;
	else
		p = max(0.0f, dot(n, wi)) / M_PIf;
	return diffuse_f(wi, wo, n);
}

// Samples a new wi direction for pathtracing and returns a simple blinnphong brdf
RT_CALLABLE_PROGRAM float3 blinnphong_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float pdf;
	float bdf_val;
	float3 tangent;
	float3 bitangent;

	// Importance sample
	create_onb(n, tangent, bitangent);
	float2 sample = make_float2(rnd(seed), rnd(seed));

	float3 wh = sample_phong_lobe(sample, Pr, tangent, bitangent, n);

	if (dot(wo, n) <= 0.0f)
		return make_float3(0.0f);

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

// Samples a new wi direction for pathtracing and returns a simple blinnphong brdf
RT_CALLABLE_PROGRAM float3 blinnphongmetal_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	return blinnphong_samplewi(seed, wi, wo, n, p);
}

// Samples a new wi direction for pathtracing and returns a brdf linearly blended between reflectivity and metalness
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