#pragma once

#include "../common/common.h"

struct BSDF;

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;
    static constexpr float RayEpsilon = 1e-4;



    template <typename RealType>
    struct CompensatedSum {
        RealType result;
        RealType comp;

        CUDA_DEVICE_FUNCTION CompensatedSum(const RealType &value) : result(value), comp(0.0) { };

        CUDA_DEVICE_FUNCTION CompensatedSum &operator=(const RealType &value) {
            result = value;
            comp = 0;
            return *this;
        }

        CUDA_DEVICE_FUNCTION CompensatedSum &operator+=(const RealType &value) {
            RealType cInput = value - comp;
            RealType sumTemp = result + cInput;
            comp = (sumTemp - result) - cInput;
            result = sumTemp;
            return *this;
        }

        CUDA_DEVICE_FUNCTION operator RealType() const { return result; };
    };

    //using FloatSum = float;
    using FloatSum = CompensatedSum<float>;



    class PCG32RNG {
        uint64_t state;

    public:
        CUDA_DEVICE_FUNCTION PCG32RNG() {}

        void setState(uint64_t _state) { state = _state; }

        CUDA_DEVICE_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_DEVICE_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        const RealType* m_PMF;
        const RealType* m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        DiscreteDistribution1DTemplate(const RealType* PMF, const RealType* CDF, RealType integral, uint32_t numValues) :
            m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

        CUDA_DEVICE_FUNCTION DiscreteDistribution1DTemplate() {}

        CUDA_DEVICE_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            *prob = m_PMF[idx];
            return idx;
        }

        CUDA_DEVICE_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
            Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
            int idx = 0;
            for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                if (idx + d >= m_numValues)
                    continue;
                if (m_CDF[idx + d] <= u)
                    idx += d;
            }
            Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
            *prob = m_PMF[idx];
            *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
            Assert(isfinite(*remapped), "Remapped value is indefinite %g.", *remapped);
            return idx;
        }

        CUDA_DEVICE_FUNCTION RealType evaluatePMF(uint32_t idx) const {
            Assert(idx >= 0 && idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
            return m_PMF[idx];
        }

        CUDA_DEVICE_FUNCTION RealType integral() const { return m_integral; }

        CUDA_DEVICE_FUNCTION uint32_t numValues() const { return m_numValues; }
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    enum RayType {
        RayType_Primary = 0,
        RayType_Visibility,
        NumRayTypes
    };



    struct Vertex {
        float3 position;
        float3 normal;
        float2 texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION float2 calcScreenPosition(const float3 &posInWorld) const {
            Matrix3x3 invOri = inverse(orientation);
            float3 posInView = invOri * (posInWorld - position);
            float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan(fovY / 2);
            float w = aspect * h;
            return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                               1 - (posAtZ1.y + 0.5f * h) / h);
        }
#endif
    };



    struct MaterialData;

    using SetupBSDF = optixu::DirectCallableProgramID<void(const MaterialData &matData, const float2 &texCoord, BSDF* bsdf)>;
    using GetBaseColor = optixu::DirectCallableProgramID<float3(const uint32_t* data, const float3 &vout)>;
    using EvaluateBSDF = optixu::DirectCallableProgramID<float3(const uint32_t* data, const float3 &vin, const float3 &vout)>;

    struct MaterialData {
        union {
            struct {
                CUtexObject reflectance;
            } asLambert;
            struct {
                CUtexObject baseColor;
                CUtexObject specular;
                CUtexObject smoothness;
            } asDiffuseAndSpecular;
        };
        float3 emittance;

        SetupBSDF setupBSDF;
        GetBaseColor getBaseColor;
        EvaluateBSDF evaluateBSDF;
    };

    struct GeometryInstanceData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
        DiscreteDistribution1D emitterPrimDist;
        uint32_t materialSlot;
        uint32_t geomInstSlot;
    };

    struct InstanceData {
        Matrix4x4 transform;
        Matrix4x4 prevTransform;
        Matrix3x3 normalMatrix;

        const uint32_t* geomInstSlots;
        uint32_t numGeomInsts;
        DiscreteDistribution1D lightGeomInstDist;
    };



    struct HitPointParams {
        uint32_t instanceSlot;
        uint32_t geomInstSlot;
        uint32_t primitiveIndex;
        float b1;
        float b2;
    };



    CUDA_DEVICE_FUNCTION float convertToWeight(const float3 &color) {
        //return sRGB_calcLuminance(color);
        return (color.x + color.y + color.z) / 3;
    }



    struct LightSample {
        float uLight;
        float uPosition[2];
    };

    template <typename SampleType, uint32_t N>
    class Reservoir {
        SampleType m_samples[N];
        FloatSum m_sumWeights;
        uint32_t m_numSamples;

    public:
        CUDA_DEVICE_FUNCTION void initialize() {
            m_sumWeights = 0;
            m_numSamples = 0;
        }
        CUDA_DEVICE_FUNCTION void update(const SampleType &newSample, float weight, float u, float uSlot) {
            m_sumWeights += weight;
            if constexpr (N > 1) {
                if (m_numSamples < N) {
                    m_samples[m_numSamples] = newSample;
                }
                else {
                    if (u < N * weight / m_sumWeights) {
                        uint32_t slot = min(static_cast<uint32_t>(uSlot * N), N - 1);
                        m_samples[slot] = newSample;
                    }
                }
            }
            else {
                (void)uSlot;
                if (u < weight / m_sumWeights)
                    m_samples[0] = newSample;
            }
            ++m_numSamples;
        }

        CUDA_DEVICE_FUNCTION LightSample getSample(uint32_t slot) const {
            if constexpr (N > 1)
                return m_samples[slot];
            else
                return m_samples[0];
        }
        CUDA_DEVICE_FUNCTION float getSumWeights() const {
            return m_sumWeights;
        }
    };



    struct PickInfo {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint32_t matSlot;
        float3 positionInWorld;
        float3 normalInWorld;
        float3 albedo;
        float3 emittance;
        unsigned int hit : 1;
    };

    
    
    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        uint32_t numAccumFrames;
        optixu::BlockBuffer2D<PCG32RNG, 1> rngBuffer;

        union ReservoirBuffer {
            optixu::BlockBuffer2D<Reservoir<LightSample, 1>, 1> n1;
            optixu::BlockBuffer2D<Reservoir<LightSample, 2>, 1> n2;
            optixu::BlockBuffer2D<Reservoir<LightSample, 4>, 1> n4;
            optixu::BlockBuffer2D<Reservoir<LightSample, 8>, 1> n8;

            ReservoirBuffer() {}
            
            template <uint32_t log2NumSamples>
            CUDA_DEVICE_FUNCTION Reservoir<LightSample, (1 << log2NumSamples)> &get(const uint2 &index) {
                if constexpr (log2NumSamples == 0)
                    return n1[index];
                else if constexpr (log2NumSamples == 1)
                    return n2[index];
                else if constexpr (log2NumSamples == 2)
                    return n4[index];
                else /*if constexpr (log2NumSamples == 3)*/
                    return n8[index];
            }
        } reservoirBuffer;
        optixu::BlockBuffer2D<HitPointParams, 1> hitPointParamsBuffer;
        unsigned int log2NumCandidateSamples : 8;
        unsigned int log2NumSamples : 2;

        const MaterialData* materialDataBuffer;
        const GeometryInstanceData* geometryInstanceDataBuffer;
        const InstanceData* instanceDataBuffer;
        DiscreteDistribution1D lightInstDist;

        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        float2* linearFlowBuffer;
        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        int2 mousePosition;
        PickInfo* pickInfo;

        unsigned int resetFlowBuffer : 1;
    };



    enum class BufferToDisplay {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        Flow,
        DenoisedBeauty,
    };
}

#define PrimaryRayPayloadSignature Shared::PCG32RNG, Shared::HitPointParams*, float3
#define VisibilityRayPayloadSignature float
