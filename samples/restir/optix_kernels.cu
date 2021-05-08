#pragma once

#include "restir_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



CUDA_DEVICE_FUNCTION float pow2(float x) {
    return x * x;
}
CUDA_DEVICE_FUNCTION float pow3(float x) {
    return x * x * x;
}
CUDA_DEVICE_FUNCTION float pow4(float x) {
    return x * x * x * x;
}
CUDA_DEVICE_FUNCTION float pow5(float x) {
    return x * x * x * x * x;
}

template <typename T>
CUDA_DEVICE_FUNCTION T lerp(const T &v0, const T &v1, float t) {
    return (1 - t) * v0 + t * v1;
}



CUDA_DEVICE_FUNCTION float3 halfVector(const float3 &a, const float3 &b) {
    return normalize(a + b);
}

CUDA_DEVICE_FUNCTION float absDot(const float3 &a, const float3 &b) {
    return std::fabs(dot(a, b));
}

CUDA_DEVICE_FUNCTION void makeCoordinateSystem(const float3 &normal, float3* tangent, float3* bitangent) {
    float sign = normal.z >= 0 ? 1 : -1;
    const float a = -1 / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    *tangent = make_float3(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    *bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

struct ReferenceFrame {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    CUDA_DEVICE_FUNCTION ReferenceFrame(const float3 &_tangent, const float3 &_bitangent, const float3 &_normal) :
        tangent(_tangent), bitangent(_bitangent), normal(_normal) {}
    CUDA_DEVICE_FUNCTION ReferenceFrame(const float3 &_normal) : normal(_normal) {
        makeCoordinateSystem(normal, &tangent, &bitangent);
    }

    CUDA_DEVICE_FUNCTION float3 toLocal(const float3 &v) const {
        return make_float3(dot(tangent, v), dot(bitangent, v), dot(normal, v));
    }
    CUDA_DEVICE_FUNCTION float3 fromLocal(const float3 &v) const {
        return make_float3(dot(make_float3(tangent.x, bitangent.x, normal.x), v),
                           dot(make_float3(tangent.y, bitangent.y, normal.y), v),
                           dot(make_float3(tangent.z, bitangent.z, normal.z), v));
    }
};



struct BSDF {
    static constexpr uint32_t NumDwords = 16;
    GetBaseColor m_getBaseColor;
    EvaluateBSDF m_evaluate;
    uint32_t m_data[NumDwords];

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        return m_getBaseColor(m_data, vout);
    }
    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vin, const float3 &vout) const {
        return m_evaluate(m_data, vin, vout);
    }
};

template <typename BSDFType>
CUDA_DEVICE_FUNCTION void setupBSDF(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf);

template <typename BSDFType>
CUDA_DEVICE_FUNCTION float3 bsdf_getBaseColor(const uint32_t* data, const float3 &vout) {
    auto &bsdf = *reinterpret_cast<const BSDFType*>(data);
    return bsdf.getBaseColor(vout);
}

template <typename BSDFType>
CUDA_DEVICE_FUNCTION float3 bsdf_evaluate(const uint32_t* data, const float3 &vin, const float3 &vout) {
    auto &bsdf = *reinterpret_cast<const BSDFType*>(data);
    return bsdf.evaluate(vin, vout);
}



class LambertBRDF {
    float3 m_reflectance;

public:
    CUDA_DEVICE_FUNCTION LambertBRDF(const float3 &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        return m_reflectance;
    }

    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vin, const float3 &vout) const {
        if (vin.z * vout.z > 0)
            return m_reflectance;
        else
            return make_float3(0.0f, 0.0f, 0.0f);
    }
};

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<LambertBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 reflectance = tex2DLod<float4>(matData.asLambert.reflectance, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bsdf->m_data);
    bsdfBody = LambertBRDF(make_float3(reflectance.x, reflectance.y, reflectance.z));
}



#define USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS
#define USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE

class DiffuseAndSpecularBRDF {
    struct GGXMicrofacetDistribution {
        float alpha_g;

        CUDA_DEVICE_FUNCTION float evaluate(const float3 &m) const {
            if (m.z <= 0.0f)
                return 0.0f;
            float temp = pow2(m.x) + pow2(m.y) + pow2(m.z * alpha_g);
            return pow2(alpha_g) / (Pi * pow2(temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const float3 &v, const float3 &m) const {
            if (dot(v, m) * v.z <= 0)
                return 0.0f;
            float temp = pow2(alpha_g) * (pow2(v.x) + pow2(v.y)) / pow2(v.z);
            return 2 / (1 + std::sqrt(1 + temp));
        }
        CUDA_DEVICE_FUNCTION float sample(const float3 &v, float u0, float u1,
                                          float3* m, float* mPDensity) const {
            // stretch view
            float3 sv = normalize(make_float3(alpha_g * v.x, alpha_g * v.y, v.z));

            // orthonormal basis
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            float3 T1 = (sv.z < 0.9999f) ?
                make_float3(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) :
                make_float3(1.0f, 0.0f, 0.0f);
            float3 T2 = make_float3(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = Pi * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            sincosf(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            *m = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            *m = normalize(make_float3(alpha_g * m->x, alpha_g * m->y, m->z));

            float D = evaluate(*m);
            *mPDensity = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::fabs(v.z);

            return D;
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const float3 &v, const float3 &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };

    float3 m_diffuseColor;
    float3 m_specularF0Color;
    float m_roughness;

public:
    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(const float3 &baseColor, const float3 &specularF0Color, float smoothness) {
        m_diffuseColor = baseColor;
        m_specularF0Color = specularF0Color;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(const float3 &baseColor, float reflectance, float smoothness, float metallic) {
        m_diffuseColor = baseColor * (1 - metallic);
        m_specularF0Color = make_float3(0.16f * pow2(reflectance) * (1 - metallic)) + baseColor * metallic;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION float3 getBaseColor(const float3 &vout) const {
        bool entering = vout.z >= 0.0f;
        float3 dirV = entering ? vout : -vout;

        float expectedCosTheta_d = dirV.z;
        float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(expectedCosTheta_d);
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float expectedDiffFGiven = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float expectedDiffFSampled = 1.0f; // ad-hoc
        float3 diffuseDHR = m_diffuseColor * expectedDiffFGiven * expectedDiffFSampled * lerp(1.0f, 1.0f / 1.51f, m_roughness);

        //float expectedOneMinusDotVH5 = oneMinusDotVN5;
        // (1 - roughness) is an ad-hoc adjustment.
        float expectedOneMinusDotVH5 = pow5(1 - dirV.z) * (1 - m_roughness);

        float3 specularDHR = lerp(m_specularF0Color, make_float3(1.0f), expectedOneMinusDotVH5);

        return min(diffuseDHR + specularDHR, make_float3(1.0f));
    }

    CUDA_DEVICE_FUNCTION float3 evaluate(const float3 &vGiven, const float3 &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        if (vSampled.z * vGiven.z <= 0)
            return make_float3(0.0f, 0.0f, 0.0f);

        bool entering = vGiven.z >= 0.0f;
        float3 dirV = entering ? vGiven : -vGiven;
        float3 dirL = entering ? vSampled : -vSampled;

        float3 m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = pow5(1 - dotLH);

        float D = ggx.evaluate(m);
#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        constexpr float F90 = 1.0f;
        float3 F = lerp(m_specularF0Color, make_float3(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        float3 specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = make_float3(0.0f);

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        float3 diffuseValue = m_diffuseColor * (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / Pi);

        float3 ret = diffuseValue + specularValue;

        return ret;
    }
};

template<>
CUDA_DEVICE_FUNCTION void setupBSDF<DiffuseAndSpecularBRDF>(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {
    float4 baseColor = tex2DLod<float4>(matData.asDiffuseAndSpecular.baseColor, texCoord.x, texCoord.y, 0.0f);
    float4 specularF0Color = tex2DLod<float4>(matData.asDiffuseAndSpecular.specular, texCoord.x, texCoord.y, 0.0f);
    float smoothness = tex2DLod<float>(matData.asDiffuseAndSpecular.smoothness, texCoord.x, texCoord.y, 0.0f);
    auto &bsdfBody = *reinterpret_cast<DiffuseAndSpecularBRDF*>(bsdf->m_data);
    bsdfBody = DiffuseAndSpecularBRDF(make_float3(baseColor.x, baseColor.y, baseColor.z),
                                      make_float3(baseColor.x, baseColor.y, baseColor.z),
                                      smoothness);
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf) {\
        bsdf->m_getBaseColor = matData.getBaseColor;\
        bsdf->m_evaluate = matData.evaluateBSDF;\
        return setupBSDF<BSDFType>(matData, texCoord, bsdf);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _getBaseColor)(const uint32_t* data, const float3 &vout) {\
        return bsdf_getBaseColor<BSDFType>(data, vout);\
    }\
    RT_CALLABLE_PROGRAM float3 RT_DC_NAME(BSDFType ## _evaluate)(const uint32_t* data, const float3 &vin, const float3 &vout) {\
        return bsdf_evaluate<BSDFType>(data, vin, vout);\
    }

DEFINE_BSDF_CALLABLES(LambertBRDF);
DEFINE_BSDF_CALLABLES(DiffuseAndSpecularBRDF);



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    GeometryInstanceData geomInstData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_RG_NAME(primary)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.rngBuffer[launchIndex];

    float x = (launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = (launchIndex.y + 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    HitPointParams* hitPointParamsPtr = &hitPointParams;
    float3 prevP = make_float3(NAN);
    optixu::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        rng, hitPointParamsPtr, prevP);

    plp.hitPointParamsBuffer[launchIndex] = hitPointParams;



    float2 curRasterPos = make_float2(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    float2 prevRasterPos =
        plp.prevCamera.calcScreenPosition(prevP)
        * make_float2(plp.imageSize.x, plp.imageSize.y);
    float2 flow = curRasterPos - prevRasterPos;
    if (plp.resetFlowBuffer || isnan(prevP.x))
        flow = make_float2(0.0f, 0.0f);
    plp.linearFlowBuffer[launchIndex.y * plp.imageSize.x + launchIndex.x] = make_float2(flow.x, flow.y/*, 0.0f, 0.0f*/);
}

CUDA_DEVICE_FUNCTION float3 sampleLight(float ul, float u0, float u1,
                                        float3* lightPosition, float3* lightNormal, float* areaPDensity) {
    float lightProb = 1.0f;

    float instProb;
    float uGeomInst;
    uint32_t instIndex = plp.lightInstDist.sample(ul, &instProb, &uGeomInst);
    lightProb *= instProb;
    const InstanceData &inst = plp.instanceDataBuffer[instIndex];

    float geomInstProb;
    float uPrim;
    uint32_t geomInstIndex = inst.geomInstSlots[inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim)];
    lightProb *= geomInstProb;
    const GeometryInstanceData &geomInst = plp.geometryInstanceDataBuffer[geomInstIndex];

    float primProb;
    uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
    lightProb *= primProb;

    const MaterialData &mat = plp.materialDataBuffer[geomInst.materialSlot];

    const Shared::Triangle &tri = geomInst.triangleBuffer[primIndex];
    const Shared::Vertex (&v)[3] = {
        geomInst.vertexBuffer[tri.index0],
        geomInst.vertexBuffer[tri.index1],
        geomInst.vertexBuffer[tri.index2]
    };

    // Uniform sampling on unit triangle
    // A Low-Distortion Map Between Triangle and Square
    float t0 = 0.5f * u0;
    float t1 = 0.5f * u1;
    float offset = t1 - t0;
    if (offset > 0)
        t1 += offset;
    else
        t0 -= offset;
    float t2 = 1 - (t0 + t1);

    *lightPosition = t0 * v[0].position + t1 * v[1].position + t2 * v[2].position;
    *lightPosition = inst.transform * *lightPosition;
    *lightNormal = cross(v[1].position - v[0].position,
                         v[2].position - v[0].position);
    float area = length(*lightNormal);
    *lightNormal = (inst.normalMatrix * *lightNormal) / area;
    area *= 0.5f;
    *areaPDensity = lightProb / area;

    return mat.emittance;
}

CUDA_DEVICE_FUNCTION float3 sampleUnshadowedContribution(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const LightSample &lightSample, float* probDensity) {
    float3 lp;
    float3 lpn;
    float3 M = sampleLight(lightSample.uLight, lightSample.uPosition[0], lightSample.uPosition[1],
                           &lp, &lpn, probDensity);

    float3 offsetOrigin = shadingPoint/* + shadingFrame.normal * epsilon*/;
    float3 shadowRayDir = lp - offsetOrigin;
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    float3 shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    float lpCos = dot(-shadowRayDir, lpn);
    float spCos = shadowRayDirLocal.z;
    float G = lpCos * spCos / dist2;

    float3 Le = lpCos > 0 ? M / Pi : make_float3(0.0f, 0.0f, 0.0f);
    float3 fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);

    float3 contribution = fsValue * Le * G;

    return contribution;
}

template <uint32_t log2NumSamples>
CUDA_DEVICE_FUNCTION void closesthit_primary_generic() {
    auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData &inst = plp.instanceDataBuffer[optixGetInstanceId()];
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    PCG32RNG rng;
    HitPointParams* hitPointParams;
    optixu::getPayloads<PrimaryRayPayloadSignature>(&rng, &hitPointParams, nullptr);

    auto hp = HitPointParameter::get();
    float3 p;
    float3 prevP;
    float3 sn;
    float2 texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        float3 localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        p = optixTransformPointFromObjectToWorldSpace(localP);
        prevP = inst.prevTransform * localP;
        sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));
    }
    optixu::setPayloads<PrimaryRayPayloadSignature>(nullptr, nullptr, &prevP);

    hitPointParams->instanceSlot = optixGetInstanceId();
    hitPointParams->geomInstSlot = geomInst.geomInstSlot;
    hitPointParams->primitiveIndex = hp.primIndex;
    hitPointParams->b1 = hp.b1;
    hitPointParams->b2 = hp.b2;

    float3 vOut = -optixGetWorldRayDirection();

    const MaterialData &mat = plp.materialDataBuffer[geomInst.materialSlot];
    BSDF bsdf;
    mat.setupBSDF(mat, texCoord, &bsdf);
    ReferenceFrame shadingFrame(sn);
    float3 vOutLocal = shadingFrame.toLocal(normalize(vOut));

    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    constexpr uint32_t numSamples = 1 << log2NumSamples;
    Reservoir<LightSample, numSamples> reservoir = plp.reservoirBuffer.get<log2NumSamples>(launchIndex);

    reservoir.initialize();
    uint32_t numCandidates = 1 << plp.log2NumCandidateSamples;
    for (int i = 0; i < numCandidates; ++i) {
        LightSample sample;
        sample.uLight = rng.getFloat0cTo1o();
        sample.uPosition[0] = rng.getFloat0cTo1o();
        sample.uPosition[1] = rng.getFloat0cTo1o();

        float probDensity;
        float3 cont = sampleUnshadowedContribution(p, vOutLocal, shadingFrame, bsdf, sample, &probDensity);
        float targetDensity = convertToWeight(cont); // unnormalized
        float weight = targetDensity / probDensity;

        reservoir.update(sample, weight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
    }

    //uint32_t linearIndex = launchIndex.y * plp.imageSize.x + launchIndex.x;
    plp.rngBuffer[launchIndex] = rng;
    plp.reservoirBuffer.get<log2NumSamples>(launchIndex) = reservoir;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(primary)() {
    if (plp.log2NumSamples == 0)
        closesthit_primary_generic<0>();
    else if (plp.log2NumSamples == 1)
        closesthit_primary_generic<1>();
    else if (plp.log2NumSamples == 2)
        closesthit_primary_generic<2>();
    else if (plp.log2NumSamples == 3)
        closesthit_primary_generic<3>();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(primary)() {
    HitPointParams* hitPointParams;
    optixu::getPayloads<PrimaryRayPayloadSignature>(nullptr, &hitPointParams, nullptr);

    hitPointParams->instanceSlot = 0xFFFFFFFF;
    hitPointParams->geomInstSlot = 0xFFFFFFFF;
    hitPointParams->primitiveIndex = 0xFFFFFFFF;
    hitPointParams->b1 = 0.0f;
    hitPointParams->b2 = 0.0f;
}



CUDA_DEVICE_FUNCTION float3 performDirectLighting(
    const float3 &shadingPoint, const float3 &vOutLocal, const ReferenceFrame &shadingFrame, const BSDF &bsdf,
    const LightSample &lightSample, float* targetDensity) {
    float3 lp;
    float3 lpn;
    float lightPDensity;
    float3 M = sampleLight(lightSample.uLight, lightSample.uPosition[0], lightSample.uPosition[1],
                           &lp, &lpn, &lightPDensity);

    float3 offsetOrigin = shadingPoint + shadingFrame.normal * RayEpsilon;
    float3 shadowRayDir = lp - offsetOrigin;
    float dist2 = sqLength(shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir /= dist;
    float3 shadowRayDirLocal = shadingFrame.toLocal(shadowRayDir);

    float lpCos = dot(-shadowRayDir, lpn);
    float spCos = shadowRayDirLocal.z;

    float visibility = 1.0f;
    optixu::trace<VisibilityRayPayloadSignature>(
        plp.travHandle,
        offsetOrigin, shadowRayDir, 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Visibility, NumRayTypes, RayType_Visibility,
        visibility);

    if (visibility > 0 && lpCos > 0) {
        float3 Le = M / Pi;
        float3 fsValue = bsdf.evaluate(vOutLocal, shadowRayDirLocal);
        float G = lpCos * spCos / dist2;
        float3 ret = fsValue * Le * G;
        *targetDensity = convertToWeight(ret);
        return ret;
    }
    else {
        *targetDensity = 0.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

template <uint32_t log2NumSamples>
CUDA_DEVICE_FUNCTION void raygen_shading_generic() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const HitPointParams &hitPointParams = plp.hitPointParamsBuffer[launchIndex];

    float3 sn = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);
    float3 contribution = make_float3(0.01f, 0.01f, 0.01f);
    if (hitPointParams.instanceSlot != 0xFFFFFFFF) {
        const InstanceData &inst = plp.instanceDataBuffer[hitPointParams.instanceSlot];
        const GeometryInstanceData &geomInst = plp.geometryInstanceDataBuffer[hitPointParams.geomInstSlot];
        const MaterialData &mat = plp.materialDataBuffer[geomInst.materialSlot];

        float3 p;
        float2 texCoord;
        {
            const Triangle &tri = geomInst.triangleBuffer[hitPointParams.primitiveIndex];
            const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
            const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
            const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
            float b1 = hitPointParams.b1;
            float b2 = hitPointParams.b2;
            float b0 = 1 - (b1 + b2);
            float3 localP = b0 * v0.position + b1 * v1.position + b2 * v2.position;
            sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
            texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

            p = inst.transform * localP;
            sn = normalize(inst.normalMatrix * sn);
        }

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(sn);
        float3 vOut = normalize(plp.camera.position - p);
        float3 vOutLocal = shadingFrame.toLocal(vOut);

        constexpr uint32_t numSamples = 1 << log2NumSamples;
        const Reservoir<LightSample, numSamples> &reservoir = plp.reservoirBuffer.get<log2NumSamples>(launchIndex);

        contribution = make_float3(0.0f);
        if (vOutLocal.z > 0)
            contribution += mat.emittance / Pi;

        uint32_t numCandidateSamples = 1 << plp.log2NumCandidateSamples;
        float3 directCont = make_float3(0.0f);
#pragma unroll
        for (int i = 0; i < numSamples; ++i) {
            const LightSample &sample = reservoir.getSample(i);
            float targetDensity;
            float3 c = performDirectLighting(p, vOutLocal, shadingFrame, bsdf, sample, &targetDensity);
            float correctionFactor = reservoir.getSumWeights() / (numCandidateSamples * targetDensity);
            if (!isfinite(correctionFactor))
                correctionFactor = 1.0f;
            directCont += correctionFactor * c;
        }
        directCont /= numSamples;

        contribution += directCont;

        albedo = bsdf.getBaseColor(vOutLocal);
    }



    // Normal input to the denoiser should be in camera space (right handed, looking down the negative Z-axis).
    float3 firstHitNormal = transpose(plp.camera.orientation) * sn;

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevAlbedoResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevNormalResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 0) {
        prevColorResult = getXYZ(plp.beautyAccumBuffer.read(launchIndex));
        prevAlbedoResult = getXYZ(plp.albedoAccumBuffer.read(launchIndex));
        prevNormalResult = getXYZ(plp.normalAccumBuffer.read(launchIndex));
    }
    float curWeight = 1.0f / (1 + plp.numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    float3 albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * albedo;
    float3 normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
    plp.beautyAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
    plp.albedoAccumBuffer.write(launchIndex, make_float4(albedoResult, 1.0f));
    plp.normalAccumBuffer.write(launchIndex, make_float4(normalResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(shading)() {
    if (plp.log2NumSamples == 0)
        raygen_shading_generic<0>();
    else if (plp.log2NumSamples == 1)
        raygen_shading_generic<1>();
    else if (plp.log2NumSamples == 2)
        raygen_shading_generic<2>();
    else if (plp.log2NumSamples == 3)
        raygen_shading_generic<3>();
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(pick)() {
    PickInfo pickInfo = {};
    pickInfo.hit = false;
    pickInfo.instSlot = 0xFFFFFFFF;
    pickInfo.geomInstSlot = 0xFFFFFFFF;
    pickInfo.matSlot = 0xFFFFFFFF;
    pickInfo.primIndex = 0xFFFFFFFF;
    pickInfo.positionInWorld = make_float3(0.0f);
    pickInfo.albedo = make_float3(0.0f);
    pickInfo.emittance = make_float3(0.0f);
    pickInfo.normalInWorld = make_float3(0.0f);

    if (plp.mousePosition.x < 0 || plp.mousePosition.x >= plp.imageSize.x ||
        plp.mousePosition.y < 0 || plp.mousePosition.y >= plp.imageSize.y)
        return;

    const HitPointParams &hitPointParams = plp.hitPointParamsBuffer[plp.mousePosition];
    if (hitPointParams.instanceSlot != 0xFFFFFFFF) {
        const InstanceData &inst = plp.instanceDataBuffer[hitPointParams.instanceSlot];
        const GeometryInstanceData &geomInst = plp.geometryInstanceDataBuffer[hitPointParams.geomInstSlot];
        const MaterialData &mat = plp.materialDataBuffer[geomInst.materialSlot];

        pickInfo.hit = true;
        pickInfo.instSlot = hitPointParams.instanceSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primitiveIndex;
        pickInfo.matSlot = geomInst.materialSlot;

        float2 texCoord;
        {
            const Triangle &tri = geomInst.triangleBuffer[hitPointParams.primitiveIndex];
            const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
            const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
            const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
            float b1 = hitPointParams.b1;
            float b2 = hitPointParams.b2;
            float b0 = 1 - (b1 + b2);
            float3 p = b0 * v0.position + b1 * v1.position + b2 * v2.position;
            float3 sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
            texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

            pickInfo.positionInWorld = inst.transform * p;
            pickInfo.normalInWorld = normalize(inst.normalMatrix * sn);
        }

        BSDF bsdf;
        mat.setupBSDF(mat, texCoord, &bsdf);
        ReferenceFrame shadingFrame(pickInfo.normalInWorld);
        float3 vOut = normalize(plp.camera.position - pickInfo.positionInWorld);
        float3 vOutLocal = shadingFrame.toLocal(normalize(vOut));

        pickInfo.albedo = bsdf.getBaseColor(vOutLocal);
        pickInfo.emittance = mat.emittance;
    }

    *plp.pickInfo = pickInfo;
}
