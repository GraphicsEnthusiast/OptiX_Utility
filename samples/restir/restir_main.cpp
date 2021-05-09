﻿/*

JP: 

EN: 

*/

#include "restir_shared.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

// Include glfw3.h after our OpenGL definitions
#include "../common/gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"



struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

KeyState g_keyForward;
KeyState g_keyBackward;
KeyState g_keyLeftward;
KeyState g_keyRightward;
KeyState g_keyUpward;
KeyState g_keyDownward;
KeyState g_keyTiltLeft;
KeyState g_keyTiltRight;
KeyState g_keyFasterPosMovSpeed;
KeyState g_keySlowerPosMovSpeed;
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;

float g_cameraPositionalMovingSpeed;
float g_cameraDirectionalMovingSpeed;
float g_cameraTiltSpeed;
Quaternion g_cameraOrientation;
Quaternion g_tempCameraOrientation;
float3 g_cameraPosition;



enum CallableProgram {
    CallableProgram_SetupLambertBRDF = 0,
    CallableProgram_LambertBRDF_getBaseColor,
    CallableProgram_LambertBRDF_evaluate,
    CallableProgram_SetupDiffuseAndSpecularBRDF,
    CallableProgram_DiffuseAndSpecularBRDF_getBaseColor,
    CallableProgram_DiffuseAndSpecularBRDF_evaluate,
    NumCallablePrograms
};

struct GPUEnvironment {
    static constexpr cudau::BufferType bufferType = cudau::BufferType::Device;
    static constexpr uint32_t maxNumMaterials = 1024;
    static constexpr uint32_t maxNumGeometryInstances = 8192;
    static constexpr uint32_t maxNumInstances = 8192;

    CUcontext cuContext;
    optixu::Context optixContext;

    optixu::Pipeline pipeline;
    optixu::Module mainModule;
    optixu::ProgramGroup primaryRayGenProgram;
    optixu::ProgramGroup shadingRayGenProgram;
    optixu::ProgramGroup pickRayGenProgram;
    optixu::ProgramGroup primaryMissProgram;
    optixu::ProgramGroup emptyMissProgram;
    optixu::ProgramGroup primaryHitProgramGroup;
    optixu::ProgramGroup visibilityHitProgramGroup;
    std::vector<optixu::ProgramGroup> callablePrograms;

    cudau::Buffer shaderBindingTable;

    optixu::Material defaultMaterial;

    optixu::Scene scene;

    SlotFinder materialSlotFinder;
    SlotFinder geomInstSlotFinder;
    SlotFinder instSlotFinder;
    cudau::TypedBuffer<Shared::MaterialData> materialDataBuffer;
    cudau::TypedBuffer<Shared::GeometryInstanceData> geomInstDataBuffer;
    cudau::TypedBuffer<Shared::InstanceData> instDataBuffer[2];

    void initialize() {
        int32_t cuDeviceCount;
        CUDADRV_CHECK(cuInit(0));
        CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
        CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
        CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

        optixContext = optixu::Context::create(cuContext);

        pipeline = optixContext.createPipeline();

        // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
        // EN: This sample uses two-level AS (single-level instancing).
        pipeline.setPipelineOptions(
            std::max({
                optixu::calcSumDwords<PrimaryRayPayloadSignature>(),
                optixu::calcSumDwords<VisibilityRayPayloadSignature>()
                     }),
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(Shared::PipelineLaunchParameters),
            false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            OPTIX_EXCEPTION_FLAG_DEBUG,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::string ptx = readTxtFile(getExecutableDirectory() / "restir/ptxes/optix_kernels.ptx");
        mainModule = pipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        primaryRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("primary"));
        shadingRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("shading"));
        pickRayGenProgram = pipeline.createRayGenProgram(
            mainModule, RT_RG_NAME_STR("pick"));
        //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
        primaryMissProgram = pipeline.createMissProgram(
            mainModule, RT_MS_NAME_STR("primary"));
        emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

        primaryHitProgramGroup = pipeline.createHitProgramGroupForBuiltinIS(
            OPTIX_PRIMITIVE_TYPE_TRIANGLE,
            mainModule, RT_CH_NAME_STR("primary"),
            emptyModule, nullptr);
        visibilityHitProgramGroup = pipeline.createHitProgramGroupForBuiltinIS(
            OPTIX_PRIMITIVE_TYPE_TRIANGLE,
            emptyModule, nullptr,
            mainModule, RT_AH_NAME_STR("visibility"));

        // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
        //pipeline.setExceptionProgram(exceptionProgram);
        pipeline.setNumMissRayTypes(Shared::NumRayTypes);
        pipeline.setMissProgram(Shared::RayType_Primary, primaryMissProgram);
        pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

        const char* entryPoints[] = {
            RT_DC_NAME_STR("setupLambertBRDF"),
            RT_DC_NAME_STR("LambertBRDF_getBaseColor"),
            RT_DC_NAME_STR("LambertBRDF_evaluate"),
            RT_DC_NAME_STR("setupDiffuseAndSpecularBRDF"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_getBaseColor"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluate"),
        };
        pipeline.setNumCallablePrograms(NumCallablePrograms);
        callablePrograms.resize(NumCallablePrograms);
        for (int i = 0; i < NumCallablePrograms; ++i) {
            optixu::ProgramGroup program = pipeline.createCallableProgramGroup(
                mainModule, entryPoints[i],
                emptyModule, nullptr);
            callablePrograms[i] = program;
            pipeline.setCallableProgram(i, program);
        }

        pipeline.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));



        size_t sbtSize;
        pipeline.generateShaderBindingTableLayout(&sbtSize);
        shaderBindingTable.initialize(cuContext, GPUEnvironment::bufferType, sbtSize, 1);
        shaderBindingTable.setMappedMemoryPersistent(true);
        pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());



        defaultMaterial = optixContext.createMaterial();
        defaultMaterial.setHitGroup(Shared::RayType_Primary, primaryHitProgramGroup);
        defaultMaterial.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);



        scene = optixContext.createScene();



        materialSlotFinder.initialize(maxNumMaterials);
        geomInstSlotFinder.initialize(maxNumGeometryInstances);
        instSlotFinder.initialize(maxNumInstances);

        materialDataBuffer.initialize(cuContext, bufferType, maxNumMaterials);
        geomInstDataBuffer.initialize(cuContext, bufferType, maxNumGeometryInstances);
        instDataBuffer[0].initialize(cuContext, bufferType, maxNumInstances);
        instDataBuffer[1].initialize(cuContext, bufferType, maxNumInstances);
    }

    void finalize() {
        instDataBuffer[1].finalize();
        instDataBuffer[0].finalize();
        geomInstDataBuffer.finalize();
        materialDataBuffer.finalize();

        instSlotFinder.finalize();
        geomInstSlotFinder.finalize();
        materialSlotFinder.finalize();

        scene.destroy();

        defaultMaterial.destroy();

        shaderBindingTable.finalize();

        for (int i = 0; i < NumCallablePrograms; ++i)
            callablePrograms[i].destroy();
        visibilityHitProgramGroup.destroy();
        primaryHitProgramGroup.destroy();
        emptyMissProgram.destroy();
        primaryMissProgram.destroy();
        pickRayGenProgram.destroy();
        shadingRayGenProgram.destroy();
        primaryRayGenProgram.destroy();
        mainModule.destroy();

        pipeline.destroy();

        optixContext.destroy();

        CUDADRV_CHECK(cuCtxDestroy(cuContext));
    }
};



template <typename RealType>
class DiscreteDistribution1DTemplate {
    cudau::TypedBuffer<RealType> m_PMF;
    cudau::TypedBuffer<RealType> m_CDF;
    RealType m_integral;
    uint32_t m_numValues;

public:
    void initialize(CUcontext cuContext, cudau::BufferType type, const RealType* values, size_t numValues) {
        m_numValues = static_cast<uint32_t>(numValues);
        m_PMF.initialize(cuContext, type, m_numValues);
        m_CDF.initialize(cuContext, type, m_numValues + 1);

        RealType* PMF = m_PMF.map();
        RealType* CDF = m_CDF.map();
        std::memcpy(PMF, values, sizeof(RealType) * m_numValues);

        Shared::CompensatedSum<RealType> sum(0);
        for (int i = 0; i < m_numValues; ++i) {
            CDF[i] = sum;
            sum += PMF[i];
        }
        m_integral = sum;
        for (int i = 0; i < m_numValues; ++i) {
            PMF[i] /= m_integral;
            CDF[i] /= m_integral;
        }
        CDF[m_numValues] = 1.0f;

        m_CDF.unmap();
        m_PMF.unmap();
    }
    void finalize() {
        if (m_CDF.isInitialized() && m_PMF.isInitialized()) {
            m_CDF.finalize();
            m_PMF.finalize();
        }
    }

    DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
        m_PMF = std::move(v.m_PMF);
        m_CDF = std::move(v.m_CDF);
        m_integral = v.m_integral;
        m_numValues = v.m_numValues;
        return *this;
    }

    RealType getIntengral() const {
        return m_integral;
    }

    void getDeviceType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
        if (m_PMF.isInitialized() && m_CDF.isInitialized())
            new (instance) Shared::DiscreteDistribution1DTemplate<RealType>(
                m_PMF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
    }
};

using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



struct Material {
    struct Lambert {
        cudau::Array reflectance;
        CUtexObject texReflectance;
    };
    struct DiffuseAndSpecular {
        cudau::Array baseColor;
        CUtexObject texBaseColor;
        cudau::Array specular;
        CUtexObject texSpecular;
        cudau::Array smoothness;
        CUtexObject texSmoothness;
    };
    std::variant<Lambert, DiffuseAndSpecular> body;
    cudau::Array normal;
    CUtexObject texNormal;
    float3 emittance;
    uint32_t materialSlot;

    Material() {}
};

struct GeometryInstance {
    const Material* mat;

    cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
    DiscreteDistribution1D emitterPrimDist;
    uint32_t geomInstSlot;
    optixu::GeometryInstance optixGeomInst;
};

struct GeometryGroup {
    std::set<const GeometryInstance*> geomInsts;

    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
};

struct Instance {
    const GeometryGroup* geomGroup;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    DiscreteDistribution1D lightGeomInstDist;
    uint32_t instSlot;
    optixu::Instance optixInst;
};



struct FlattenedNode {
    Matrix4x4 transform;
    std::vector<uint32_t> meshIndices;
};

void computeFlattenedNodes(const aiScene* scene, const Matrix4x4 &parentXfm, const aiNode* curNode,
                           std::vector<FlattenedNode> &flattenedNodes) {
    aiMatrix4x4 curAiXfm = curNode->mTransformation;
    Matrix4x4 curXfm = Matrix4x4(float4(curAiXfm.a1, curAiXfm.a2, curAiXfm.a3, curAiXfm.a4),
                                 float4(curAiXfm.b1, curAiXfm.b2, curAiXfm.b3, curAiXfm.b4),
                                 float4(curAiXfm.c1, curAiXfm.c2, curAiXfm.c3, curAiXfm.c4),
                                 float4(curAiXfm.d1, curAiXfm.d2, curAiXfm.d3, curAiXfm.d4));
    FlattenedNode flattenedNode;
    flattenedNode.transform = parentXfm * curXfm;
    flattenedNode.meshIndices.resize(curNode->mNumMeshes);
    std::copy_n(curNode->mMeshes, curNode->mNumMeshes, flattenedNode.meshIndices.data());
    flattenedNodes.push_back(flattenedNode);

    for (int cIdx = 0; cIdx < curNode->mNumChildren; ++cIdx)
        computeFlattenedNodes(scene, flattenedNode.transform, curNode->mChildren[cIdx], flattenedNodes);
}

Material* createLambertMaterial(
    GPUEnvironment &gpuEnv,
    const std::filesystem::path &reflectancePath, const float3 &immReflectance,
    const float3 &emittance) {
    Shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();

    cudau::TextureSampler sampler_sRGB;
    sampler_sRGB.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_sRGB.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::TextureSampler sampler;
    sampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
    sampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
    sampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);

    Material* mat = new Material();

    mat->body = Material::Lambert();
    auto &body = std::get<Material::Lambert>(mat->body);
    if (reflectancePath.empty()) {
        uint32_t data = ((std::min<uint32_t>(255 * immReflectance.x, 255) << 0) |
                         (std::min<uint32_t>(255 * immReflectance.y, 255) << 8) |
                         (std::min<uint32_t>(255 * immReflectance.z, 255) << 16) |
                         255 << 24);
        body.reflectance.initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        body.reflectance.write<uint8_t>(reinterpret_cast<uint8_t*>(&data), 4);
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(reflectancePath.string().c_str(),
                                             &width, &height, &n, 4);
        body.reflectance.initialize2D(
            gpuEnv.cuContext, cudau::ArrayElementType::UInt8, 4,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        body.reflectance.write<uint8_t>(linearImageData, width * height * 4);
        stbi_image_free(linearImageData);
    }
    body.texReflectance = sampler_sRGB.createTextureObject(body.reflectance);

    mat->emittance = emittance;

    mat->materialSlot = gpuEnv.materialSlotFinder.getFirstAvailableSlot();
    gpuEnv.materialSlotFinder.setInUse(mat->materialSlot);

    Shared::MaterialData matData = {};
    matData.asLambert.reflectance = body.texReflectance;
    matData.emittance = mat->emittance;
    matData.setupBSDF = Shared::SetupBSDF(CallableProgram_SetupLambertBRDF);
    matData.getBaseColor = Shared::GetBaseColor(CallableProgram_LambertBRDF_getBaseColor);
    matData.evaluateBSDF = Shared::EvaluateBSDF(CallableProgram_LambertBRDF_evaluate);
    matDataOnHost[mat->materialSlot] = matData;

    return mat;
}

GeometryInstance* createGeometryInstance(
    GPUEnvironment &gpuEnv,
    const std::vector<Shared::Vertex> &vertices,
    const std::vector<Shared::Triangle> &triangles,
    const Material* mat) {
    Shared::GeometryInstanceData* geomInstDataOnHost = gpuEnv.geomInstDataBuffer.getMappedPointer();

    std::vector<float> emitterImportances(triangles.size());
    float lumEmittance = Shared::convertToWeight(mat->emittance);
    for (int triIdx = 0; triIdx < emitterImportances.size(); ++triIdx) {
        const Shared::Triangle &tri = triangles[triIdx];
        const Shared::Vertex(&vs)[3] = {
            vertices[tri.index0],
            vertices[tri.index1],
            vertices[tri.index2],
        };
        float area = 0.5f * length(cross(vs[2].position - vs[0].position,
                                         vs[1].position - vs[0].position));
        Assert(area >= 0.0f, "Area must be positive.");
        emitterImportances[triIdx] = lumEmittance * area;
    }

    GeometryInstance* geomInst = new GeometryInstance();
    geomInst->mat = mat;
    geomInst->vertexBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, vertices);
    geomInst->triangleBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, triangles);
    geomInst->emitterPrimDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                         emitterImportances.data(), emitterImportances.size());
    geomInst->geomInstSlot = gpuEnv.geomInstSlotFinder.getFirstAvailableSlot();
    gpuEnv.geomInstSlotFinder.setInUse(geomInst->geomInstSlot);

    Shared::GeometryInstanceData geomInstData = {};
    geomInstData.vertexBuffer = geomInst->vertexBuffer.getDevicePointer();
    geomInstData.triangleBuffer = geomInst->triangleBuffer.getDevicePointer();
    geomInst->emitterPrimDist.getDeviceType(&geomInstData.emitterPrimDist);
    geomInstData.materialSlot = mat->materialSlot;
    geomInstData.geomInstSlot = geomInst->geomInstSlot;
    geomInstDataOnHost[geomInst->geomInstSlot] = geomInstData;

    geomInst->optixGeomInst = gpuEnv.scene.createGeometryInstance();
    geomInst->optixGeomInst.setVertexBuffer(geomInst->vertexBuffer);
    geomInst->optixGeomInst.setTriangleBuffer(geomInst->triangleBuffer);
    geomInst->optixGeomInst.setNumMaterials(1, optixu::BufferView());
    geomInst->optixGeomInst.setMaterial(0, 0, gpuEnv.defaultMaterial);
    geomInst->optixGeomInst.setUserData(geomInstData);

    return geomInst;
}

GeometryGroup* createGeometryGroup(
    GPUEnvironment &gpuEnv,
    const std::set<const GeometryInstance*> &geomInsts) {
    GeometryGroup* geomGroup = new GeometryGroup();
    geomGroup->geomInsts = geomInsts;

    geomGroup->optixGas = gpuEnv.scene.createGeometryAccelerationStructure();
    for (auto it = geomInsts.cbegin(); it != geomInsts.cend(); ++it)
        geomGroup->optixGas.addChild((*it)->optixGeomInst);
    geomGroup->optixGas.setNumMaterialSets(1);
    geomGroup->optixGas.setNumRayTypes(0, Shared::NumRayTypes);

    return geomGroup;
}

Instance* createInstance(
    GPUEnvironment &gpuEnv,
    const GeometryGroup* geomGroup,
    const Matrix4x4 &transform) {
    Shared::InstanceData* instDataOnHost = gpuEnv.instDataBuffer[0].getMappedPointer();

    std::vector<uint32_t> geomInstSlots;
    std::vector<float> lightImportances;
    for (auto it = geomGroup->geomInsts.cbegin(); it != geomGroup->geomInsts.cend(); ++it) {
        const GeometryInstance* geomInst = *it;
        geomInstSlots.push_back(geomInst->geomInstSlot);
        lightImportances.push_back(geomInst->emitterPrimDist.getIntengral());
    }

    Instance* inst = new Instance();
    inst->geomGroup = geomGroup;
    inst->geomInstSlots.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, geomInstSlots);
    inst->lightGeomInstDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                       lightImportances.data(), lightImportances.size());
    inst->instSlot = gpuEnv.instSlotFinder.getFirstAvailableSlot();
    gpuEnv.instSlotFinder.setInUse(inst->instSlot);

    Shared::InstanceData instData = {};
    instData.transform = transform;
    instData.prevTransform = transform;
    instData.normalMatrix = transpose(inverse(transform.getUpperLeftMatrix()));
    instData.geomInstSlots = inst->geomInstSlots.getDevicePointer();
    instData.numGeomInsts = inst->geomInstSlots.numElements();
    inst->lightGeomInstDist.getDeviceType(&instData.lightGeomInstDist);
    instDataOnHost[inst->instSlot] = instData;

    inst->optixInst = gpuEnv.scene.createInstance();
    inst->optixInst.setID(inst->instSlot);
    inst->optixInst.setChild(geomGroup->optixGas);
    float xfm[12] = {
        transform.m00, transform.m01, transform.m02, transform.m03,
        transform.m10, transform.m11, transform.m12, transform.m13,
        transform.m20, transform.m21, transform.m22, transform.m23,
    };
    inst->optixInst.setTransform(xfm);

    return inst;
}

void createTriangleMeshes(const std::filesystem::path &filePath,
                          const Matrix4x4 &rootTransform,
                          GPUEnvironment &gpuEnv,
                          std::vector<Material*> &materials,
                          std::vector<GeometryInstance*> &geomInsts,
                          std::vector<GeometryGroup*> &geomGroups,
                          std::vector<Instance*> &insts) {
    hpprintf("Reading: %s ... ", filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath.string(),
                                             aiProcess_Triangulate |
                                             aiProcess_GenNormals |
                                             aiProcess_FlipUVs);
    if (!scene) {
        hpprintf("Failed to load %s.\n", filePath.string().c_str());
        return;
    }
    hpprintf("done.\n", filePath.string().c_str());

    std::filesystem::path dirPath = filePath;
    dirPath.remove_filename();

    materials.clear();
    Shared::MaterialData* matDataOnHost = gpuEnv.materialDataBuffer.getMappedPointer();
    for (int matIdx = 0; matIdx < scene->mNumMaterials; ++matIdx) {
        std::filesystem::path reflectancePath;
        float3 immReflectance;
        float3 immEmittance;

        const aiMaterial* aiMat = scene->mMaterials[matIdx];
        aiString strValue;
        float color[3];

        if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
            reflectancePath = dirPath / strValue.C_Str();
        }
        else {
            if (!aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
                color[0] = 1.0f;
                color[1] = 0.0f;
                color[2] = 1.0f;
            }
            immReflectance = float3(color[0], color[1], color[2]);
        }

        if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS)
            immEmittance = float3(color[0], color[1], color[2]);

        materials.push_back(createLambertMaterial(gpuEnv, reflectancePath, immReflectance, immEmittance));
    }

    geomInsts.clear();
    for (int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        const aiMesh* aiMesh = scene->mMeshes[meshIdx];

        std::vector<Shared::Vertex> vertices(aiMesh->mNumVertices);
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            const aiVector3D &aip = aiMesh->mVertices[vIdx];
            const aiVector3D &ain = aiMesh->mNormals[vIdx];
            const aiVector3D &ait = aiMesh->mTextureCoords[0][vIdx];

            Shared::Vertex v;
            v.position = float3(aip.x, aip.y, aip.z);
            v.normal = float3(ain.x, ain.y, ain.z);
            v.texCoord = float2(ait.x, ait.y);
            vertices[vIdx] = v;
        }

        std::vector<Shared::Triangle> triangles(aiMesh->mNumFaces);
        for (int fIdx = 0; fIdx < triangles.size(); ++fIdx) {
            const aiFace &aif = aiMesh->mFaces[fIdx];
            Assert(aif.mNumIndices == 3, "Number of face vertices must be 3 here.");
            Shared::Triangle tri;
            tri.index0 = aif.mIndices[0];
            tri.index1 = aif.mIndices[1];
            tri.index2 = aif.mIndices[2];
            triangles[fIdx] = tri;
        }

        geomInsts.push_back(createGeometryInstance(gpuEnv, vertices, triangles, materials[aiMesh->mMaterialIndex]));
    }

    std::vector<FlattenedNode> flattenedNodes;
    computeFlattenedNodes(scene, rootTransform, scene->mRootNode, flattenedNodes);

    geomGroups.clear();
    insts.clear();
    Shared::InstanceData* instDataOnHost = gpuEnv.instDataBuffer[0].getMappedPointer();
    std::map<std::set<const GeometryInstance*>, GeometryGroup*> geomGroupMap;
    for (int nodeIdx = 0; nodeIdx < flattenedNodes.size(); ++nodeIdx) {
        const FlattenedNode &node = flattenedNodes[nodeIdx];
        if (node.meshIndices.size() == 0)
            continue;

        std::set<const GeometryInstance*> srcGeomInsts;
        for (int i = 0; i < node.meshIndices.size(); ++i)
            srcGeomInsts.insert(geomInsts[node.meshIndices[i]]);
        GeometryGroup* geomGroup;
        if (geomGroupMap.count(srcGeomInsts) > 0) {
            geomGroup = geomGroupMap.at(srcGeomInsts);
        }
        else {
            geomGroup = createGeometryGroup(gpuEnv, srcGeomInsts);
            geomGroups.push_back(geomGroup);
        }

        insts.push_back(createInstance(gpuEnv, geomGroup, node.transform));
    }
}

void createRectangleLight(float width, float depth, const float3 &emittance,
                          const Matrix4x4 &transform,
                          GPUEnvironment &gpuEnv,
                          Material** material,
                          GeometryInstance** geomInst,
                          GeometryGroup** geomGroup,
                          Instance** inst) {
    *material = createLambertMaterial(gpuEnv, "", float3(0.8f), emittance);

    std::vector<Shared::Vertex> vertices = {
        Shared::Vertex{float3(-0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float2(0.0f, 1.0f)},
        Shared::Vertex{float3(0.5f * width, 0.0f, -0.5f * depth), float3(0, -1, 0), float2(1.0f, 1.0f)},
        Shared::Vertex{float3(0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float2(1.0f, 0.0f)},
        Shared::Vertex{float3(-0.5f * width, 0.0f, 0.5f * depth), float3(0, -1, 0), float2(0.0f, 0.0f)},
    };
    std::vector<Shared::Triangle> triangles = {
        Shared::Triangle{0, 1, 2},
        Shared::Triangle{0, 2, 3},
    };
    *geomInst = createGeometryInstance(gpuEnv, vertices, triangles, *material);

    std::set<const GeometryInstance*> srcGeomInsts = { *geomInst };
    *geomGroup = createGeometryGroup(gpuEnv, srcGeomInsts);

    *inst = createInstance(gpuEnv, *geomGroup, transform);
}



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



namespace ImGui {
    template <typename EnumType>
    bool RadioButtonE(const char* label, EnumType* v, EnumType v_button) {
        return RadioButton(label, reinterpret_cast<int*>(v), static_cast<int>(v_button));
    }
}

int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path exeDir = getExecutableDirectory();

    bool takeScreenShot = false;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot")
            takeScreenShot = true;
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    constexpr bool enableGLDebugCallback = DEBUG_SELECT(true, false);

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if constexpr (enableGLDebugCallback)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    int32_t renderTargetSizeX = 1024;
    int32_t renderTargetSizeY = 1024;

    // JP: ウインドウの初期化。
    //     HiDPIディスプレイに対応する。
    // EN: Initialize a window.
    //     Support Hi-DPI display.
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
    float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow(static_cast<int32_t>(renderTargetSizeX * UIScaling),
                                          static_cast<int32_t>(renderTargetSizeY * UIScaling),
                                          "OptiX Utility - ReSTIR", NULL, NULL);
    glfwSetWindowUserPointer(window, nullptr);
    if (!window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    int32_t curFBWidth;
    int32_t curFBHeight;
    glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    if constexpr (enableGLDebugCallback) {
        glu::enableDebugCallback(true);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
    }

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGuiStyle guiStyle, guiStyleWithGamma;
    ImGui::StyleColorsDark(&guiStyle);
    guiStyleWithGamma = guiStyle;
    const auto degamma = [](const ImVec4 &color) {
        return ImVec4(sRGB_degamma_s(color.x),
                      sRGB_degamma_s(color.y),
                      sRGB_degamma_s(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        guiStyleWithGamma.Colors[i] = degamma(guiStyleWithGamma.Colors[i]);
    }
    ImGui::GetStyle() = guiStyleWithGamma;

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

        switch (button) {
        case GLFW_MOUSE_BUTTON_MIDDLE: {
            devPrintf("Mouse Middle\n");
            g_buttonRotate.recordStateChange(action == GLFW_PRESS, frameIndex);
            break;
        }
        default:
            break;
        }
                               });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        g_mouseX = x;
        g_mouseY = y;
                             });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        switch (key) {
        case GLFW_KEY_W: {
            g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_S: {
            g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_A: {
            g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_D: {
            g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_R: {
            g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_F: {
            g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_Q: {
            g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_E: {
            g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_T: {
            g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_G: {
            g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        default:
            break;
        }
                       });

    g_cameraPositionalMovingSpeed = 0.01f;
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;
    g_cameraPosition = float3(-6.4f, 3, 0);
    g_cameraOrientation = qRotateY(M_PI / 2);

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    GPUEnvironment gpuEnv;
    gpuEnv.initialize();

    CUstream cuStream;
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    gpuEnv.materialDataBuffer.map();
    gpuEnv.geomInstDataBuffer.map();
    gpuEnv.instDataBuffer[0].map();

    std::vector<Material*> materials;
    std::vector<GeometryInstance*> geomInsts;
    std::vector<GeometryGroup*> geomGroups;
    std::vector<Instance*> insts;
    {
        std::vector<Material*> tempMaterials;
        std::vector<GeometryInstance*> tempGeomInsts;
        std::vector<GeometryGroup*> tempGeomGroups;
        std::vector<Instance*> tempInsts;
        createTriangleMeshes("../../../assets/crytek_sponza/sponza.obj",
                             translate4x4(0, 0, 0.36125f) * scale4x4(0.01f),
                             gpuEnv,
                             tempMaterials,
                             tempGeomInsts,
                             tempGeomGroups,
                             tempInsts);

        materials.insert(materials.end(), tempMaterials.begin(), tempMaterials.end());
        geomInsts.insert(geomInsts.end(), tempGeomInsts.begin(), tempGeomInsts.end());
        geomGroups.insert(geomGroups.end(), tempGeomGroups.begin(), tempGeomGroups.end());
        insts.insert(insts.end(), tempInsts.begin(), tempInsts.end());
    }
    {
        Material* tempMaterial;
        GeometryInstance* tempGeomInst;
        GeometryGroup* tempGeomGroup;
        Instance* tempInst;
        createRectangleLight(2.5f, 2.5f, float3(500, 0, 0), translate4x4(5, 10, 0),
                             gpuEnv,
                             &tempMaterial,
                             &tempGeomInst,
                             &tempGeomGroup,
                             &tempInst);

        materials.push_back(tempMaterial);
        geomInsts.push_back(tempGeomInst);
        geomGroups.push_back(tempGeomGroup);
        insts.push_back(tempInst);
    }
    {
        Material* tempMaterial;
        GeometryInstance* tempGeomInst;
        GeometryGroup* tempGeomGroup;
        Instance* tempInst;
        createRectangleLight(2.5f, 2.5f, float3(0, 500, 0), translate4x4(0, 10, 0),
                             gpuEnv,
                             &tempMaterial,
                             &tempGeomInst,
                             &tempGeomGroup,
                             &tempInst);

        materials.push_back(tempMaterial);
        geomInsts.push_back(tempGeomInst);
        geomGroups.push_back(tempGeomGroup);
        insts.push_back(tempInst);
    }
    {
        Material* tempMaterial;
        GeometryInstance* tempGeomInst;
        GeometryGroup* tempGeomGroup;
        Instance* tempInst;
        createRectangleLight(2.5f, 2.5f, float3(0, 0, 500), translate4x4(-5, 10, 0),
                             gpuEnv,
                             &tempMaterial,
                             &tempGeomInst,
                             &tempGeomGroup,
                             &tempInst);

        materials.push_back(tempMaterial);
        geomInsts.push_back(tempGeomInst);
        geomGroups.push_back(tempGeomGroup);
        insts.push_back(tempInst);
    }

    gpuEnv.instDataBuffer[0].unmap();
    gpuEnv.geomInstDataBuffer.unmap();
    gpuEnv.materialDataBuffer.unmap();

    CUDADRV_CHECK(cuMemcpyDtoD(gpuEnv.instDataBuffer[1].getCUdeviceptr(),
                               gpuEnv.instDataBuffer[0].getCUdeviceptr(),
                               gpuEnv.instDataBuffer[1].sizeInBytes()));



    optixu::InstanceAccelerationStructure ias = gpuEnv.scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> iasInstanceBuffer;
    for (int i = 0; i < insts.size(); ++i) {
        const Instance* inst = insts[i];
        ias.addChild(inst->optixInst);
    }

    OptixAccelBufferSizes asSizes;
    size_t asScratchSize = 0;
    for (int i = 0; i < geomGroups.size(); ++i) {
        GeometryGroup* geomGroup = geomGroups[i];
        geomGroup->optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            false, false, false);
        geomGroup->optixGas.prepareForBuild(&asSizes);
        geomGroup->optixGasMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asSizes.outputSizeInBytes, 1);
        asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
    }

    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    ias.prepareForBuild(&asSizes);
    iasMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asSizes.outputSizeInBytes, 1);
    asScratchSize = std::max(asSizes.tempSizeInBytes, asScratchSize);
    iasInstanceBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, ias.getNumChildren());

    cudau::Buffer asScratchMem;
    asScratchMem.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, asScratchSize, 1);

    for (int i = 0; i < geomGroups.size(); ++i) {
        const GeometryGroup* geomGroup = geomGroups[i];
        geomGroup->optixGas.rebuild(cuStream, geomGroup->optixGasMem, asScratchMem);
    }

    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    gpuEnv.scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, iasInstanceBuffer, iasMem, asScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    std::vector<float> lightImportances(insts.size());
    for (int i = 0; i < insts.size(); ++i) {
        const Instance* inst = insts[i];
        lightImportances[i] = inst->lightGeomInstDist.getIntengral();
    }
    DiscreteDistribution1D lightInstDist;
    lightInstDist.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                             lightImportances.data(), lightImportances.size());
    Assert(lightInstDist.getIntengral() > 0, "No lights!");

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------

    optixu::HostBlockBuffer2D<Shared::Reservoir<Shared::LightSample, 1>, 1> reservoirBuffer_n1;
    optixu::HostBlockBuffer2D<Shared::Reservoir<Shared::LightSample, 2>, 1> reservoirBuffer_n2;
    optixu::HostBlockBuffer2D<Shared::Reservoir<Shared::LightSample, 4>, 1> reservoirBuffer_n4;
    optixu::HostBlockBuffer2D<Shared::Reservoir<Shared::LightSample, 8>, 1> reservoirBuffer_n8;

    reservoirBuffer_n1.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX, renderTargetSizeY);
    reservoirBuffer_n2.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX, renderTargetSizeY);
    reservoirBuffer_n4.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX, renderTargetSizeY);
    reservoirBuffer_n8.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX, renderTargetSizeY);

    optixu::HostBlockBuffer2D<Shared::HitPointParams, 1> hitPointParamsBuffer;
    hitPointParamsBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                    renderTargetSizeX, renderTargetSizeY);
    
    cudau::Array beautyAccumBuffer;
    cudau::Array albedoAccumBuffer;
    cudau::Array normalAccumBuffer;
    beautyAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    albedoAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    normalAccumBuffer.initialize2D(gpuEnv.cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    cudau::TypedBuffer<float4> linearBeautyBuffer;
    cudau::TypedBuffer<float4> linearAlbedoBuffer;
    cudau::TypedBuffer<float4> linearNormalBuffer;
    cudau::TypedBuffer<float2> linearFlowBuffer;
    cudau::TypedBuffer<float4> linearDenoisedBeautyBuffer;
    linearBeautyBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX * renderTargetSizeY);
    linearAlbedoBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX * renderTargetSizeY);
    linearNormalBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                  renderTargetSizeX * renderTargetSizeY);
    linearFlowBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                renderTargetSizeX * renderTargetSizeY);
    linearDenoisedBeautyBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                          renderTargetSizeX * renderTargetSizeY);

    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> rngBuffer;
    rngBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, renderTargetSizeX, renderTargetSizeY);
    {
        std::mt19937_64 rngSeed(591842031321323413);

        rngBuffer.map();
        for (int y = 0; y < renderTargetSizeY; ++y) {
            for (int x = 0; x < renderTargetSizeX; ++x) {
                Shared::PCG32RNG &rng = rngBuffer(x, y);
                rng.setState(rngSeed());
            }
        }
        rngBuffer.unmap();
    };

    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: デノイザーのセットアップ。
    //     Temporalデノイザーを使用する。
    // EN: Setup a denoiser.
    //     Use the temporal denoiser.

    constexpr bool useTiledDenoising = false; // Change this to true to use tiled denoising.
    constexpr uint32_t tileWidth = useTiledDenoising ? 256 : 0;
    constexpr uint32_t tileHeight = useTiledDenoising ? 256 : 0;
    optixu::Denoiser denoiser = gpuEnv.optixContext.createDenoiser(OPTIX_DENOISER_MODEL_KIND_TEMPORAL, true, true);
    size_t stateSize;
    size_t scratchSize;
    size_t scratchSizeForComputeIntensity;
    uint32_t numTasks;
    denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                     &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                     &numTasks);
    hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
    hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
    hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    denoiserStateBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, stateSize, 1);
    denoiserScratchBuffer.initialize(gpuEnv.cuContext, GPUEnvironment::bufferType,
                                     std::max(scratchSize, scratchSizeForComputeIntensity), 1);

    std::vector<optixu::DenoisingTask> denoisingTasks(numTasks);
    denoiser.getTasks(denoisingTasks.data());

    denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(&moduleCopyBuffers, (getExecutableDirectory() / "restir/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyToLinearBuffers(moduleCopyBuffers, "copyToLinearBuffers", cudau::dim3(8, 8), 0);
    cudau::Kernel kernelVisualizeToOutputBuffer(moduleCopyBuffers, "visualizeToOutputBuffer", cudau::dim3(8, 8), 0);

    CUdeviceptr hdrIntensity;
    CUDADRV_CHECK(cuMemAlloc(&hdrIntensity, sizeof(float)));

    // END: Setup a denoiser.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    glu::Texture2D outputTexture;
    cudau::Array outputArray;
    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
    outputArray.initializeFromGLTexture2D(gpuEnv.cuContext, outputTexture.getHandle(),
                                          cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    outputBufferSurfaceHolder.initialize(&outputArray);

    glu::Sampler outputSampler;
    outputSampler.initialize(glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
                             glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    glu::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    glu::GraphicsProgram drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "restir/shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "restir/shaders/drawOptiXResult.frag"));



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    //plp.numAccumFrames = ;
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();

    plp.reservoirBuffer.n1 = reservoirBuffer_n1.getBlockBuffer2D();
    plp.hitPointParamsBuffer = hitPointParamsBuffer.getBlockBuffer2D();
    plp.log2NumCandidateSamples = 4;
    plp.log2NumSamples = 0;

    plp.materialDataBuffer = gpuEnv.materialDataBuffer.getDevicePointer();
    plp.geometryInstanceDataBuffer = gpuEnv.geomInstDataBuffer.getDevicePointer();
    //plp.instanceDataBuffer = ;
    lightInstDist.getDeviceType(&plp.lightInstDist);

    plp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
    plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
    plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
    plp.linearFlowBuffer = linearFlowBuffer.getDevicePointer();
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = g_cameraPosition;
    plp.camera.orientation = g_cameraOrientation.toMatrix3x3();
    plp.prevCamera = plp.camera;
    //plp.resetFlowBuffer = ;

    gpuEnv.pipeline.setScene(gpuEnv.scene);
    gpuEnv.pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    Shared::PickInfo initPickInfo = {};
    initPickInfo.hit = false;
    initPickInfo.instSlot = 0xFFFFFFFF;
    initPickInfo.geomInstSlot = 0xFFFFFFFF;
    initPickInfo.matSlot = 0xFFFFFFFF;
    initPickInfo.primIndex = 0xFFFFFFFF;
    initPickInfo.positionInWorld = make_float3(0.0f);
    initPickInfo.albedo = make_float3(0.0f);
    initPickInfo.emittance = make_float3(0.0f);
    initPickInfo.normalInWorld = make_float3(0.0f);
    cudau::TypedBuffer<Shared::PickInfo> pickInfos[2];
    pickInfos[0].initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, 1, initPickInfo);
    pickInfos[1].initialize(gpuEnv.cuContext, GPUEnvironment::bufferType, 1, initPickInfo);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer render;
        cudau::Timer denoise;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            render.initialize(context);
            denoise.initialize(context);
        }
        void finalize() {
            denoise.finalize();
            render.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    GPUTimer gpuTimers[2];
    gpuTimers[0].initialize(gpuEnv.cuContext);
    gpuTimers[1].initialize(gpuEnv.cuContext);
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    while (true) {
        uint32_t bufferIndex = frameIndex % 2;

        GPUTimer &curGPUTimer = gpuTimers[bufferIndex];

        plp.prevCamera = plp.camera;

        if (glfwWindowShouldClose(window))
            break;
        glfwPollEvents();

        bool resized = false;
        int32_t newFBWidth;
        int32_t newFBHeight;
        glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
        if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
            curFBWidth = newFBWidth;
            curFBHeight = newFBHeight;

            renderTargetSizeX = curFBWidth / UIScaling;
            renderTargetSizeY = curFBHeight / UIScaling;
            requestedSize[0] = renderTargetSizeX;
            requestedSize[1] = renderTargetSizeY;

            beautyAccumBuffer.resize(renderTargetSizeX, renderTargetSizeY);
            albedoAccumBuffer.resize(renderTargetSizeX, renderTargetSizeY);
            normalAccumBuffer.resize(renderTargetSizeX, renderTargetSizeY);
            linearBeautyBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            linearAlbedoBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            linearNormalBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            linearFlowBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            linearDenoisedBeautyBuffer.resize(renderTargetSizeX * renderTargetSizeY);

            rngBuffer.resize(renderTargetSizeX, renderTargetSizeY);
            {
                std::mt19937_64 rng(591842031321323413);

                rngBuffer.map();
                for (int y = 0; y < renderTargetSizeY; ++y)
                    for (int x = 0; x < renderTargetSizeX; ++x)
                        rngBuffer(x, y).setState(rng());
                rngBuffer.unmap();
            };

            plp.rngBuffer = rngBuffer.getBlockBuffer2D();
            plp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
            plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
            plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
            plp.linearFlowBuffer = linearFlowBuffer.getDevicePointer();

            {
                size_t stateSize;
                size_t scratchSize;
                size_t scratchSizeForComputeIntensity;
                uint32_t numTasks;
                denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                                 &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                                 &numTasks);
                hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
                hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
                hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
                denoiserStateBuffer.resize(stateSize, 1);
                denoiserScratchBuffer.resize(std::max(scratchSize, scratchSizeForComputeIntensity), 1);

                denoisingTasks.resize(numTasks);
                denoiser.getTasks(denoisingTasks.data());

                denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);
            }

            outputTexture.finalize();
            outputArray.finalize();
            outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            outputArray.initializeFromGLTexture2D(gpuEnv.cuContext, outputTexture.getHandle(),
                                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            // EN: update the pipeline parameters.
            plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        bool operatingCamera;
        bool cameraIsActuallyMoving;
        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState& a, const KeyState& b) {
                int32_t dir = 0;
                if (a.getState() == true) {
                    if (b.getState() == true)
                        dir = 0;
                    else
                        dir = 1;
                }
                else {
                    if (b.getState() == true)
                        dir = -1;
                    else
                        dir = 0;
                }
                return dir;
            };

            int32_t trackZ = decideDirection(g_keyForward, g_keyBackward);
            int32_t trackX = decideDirection(g_keyLeftward, g_keyRightward);
            int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
            int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

            g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            g_cameraPositionalMovingSpeed = std::clamp(g_cameraPositionalMovingSpeed, 1e-6f, 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
            if (g_buttonRotate.getState() == true) {
                if (g_buttonRotate.getTime() == frameIndex) {
                    lastX = g_mouseX;
                    lastY = g_mouseY;
                }
                else {
                    deltaX = g_mouseX - lastX;
                    deltaY = g_mouseY - lastY;
                }
            }

            float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            float3 axis = float3(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = float3(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation = g_cameraOrientation * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition += g_tempCameraOrientation.toMatrix3x3() * (g_cameraPositionalMovingSpeed * float3(trackX, trackY, trackZ));
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == frameIndex) {
                g_cameraOrientation = g_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                               g_keyLeftward.getState() || g_keyRightward.getState() ||
                               g_keyUpward.getState() || g_keyDownward.getState() ||
                               g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                               g_buttonRotate.getState());
            cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 ||
                                      tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;

            plp.camera.position = g_cameraPosition;
            plp.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        // Camera Window
        {
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&plp.camera.position));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / M_PI;
            rollPitchYaw[1] *= 180 / M_PI;
            rollPitchYaw[2] *= 180 / M_PI;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw, 3))
                g_cameraOrientation = qFromEulerAngles(rollPitchYaw[0] * M_PI / 180,
                                                       rollPitchYaw[1] * M_PI / 180,
                                                       rollPitchYaw[2] * M_PI / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);

            ImGui::End();
        }

        static bool useTemporalDenosier = true;
        static Shared::BufferToDisplay bufferTypeToDisplay = Shared::BufferToDisplay::DenoisedBeauty;
        static float motionVectorScale = -1.0f;
        static bool animate = false;
        bool lastFrameWasAnimated = false;
        static int32_t log2NumCandidateSamples = 4;
        static int32_t log2NumSamples = 0;
        {
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::Button(animate ? "Stop" : "Play")) {
                if (animate)
                    lastFrameWasAnimated = true;
                animate = !animate;
            }

            ImGui::Separator();
            ImGui::Text("Cursor Info:");
            Shared::PickInfo pickInfoOnHost;
            pickInfos[bufferIndex].read(&pickInfoOnHost, 1, cuStream);
            ImGui::Text("Hit: %s", pickInfoOnHost.hit ? "True" : "False");
            ImGui::Text("Instance: %u", pickInfoOnHost.instSlot);
            ImGui::Text("Geometry Instance: %u", pickInfoOnHost.geomInstSlot);
            ImGui::Text("Primitive Index: %u", pickInfoOnHost.primIndex);
            ImGui::Text("Material: %u", pickInfoOnHost.matSlot);
            ImGui::Text("Position: %.3f, %.3f, %.3f",
                        pickInfoOnHost.positionInWorld.x,
                        pickInfoOnHost.positionInWorld.y,
                        pickInfoOnHost.positionInWorld.z);
            ImGui::Text("Normal: %.3f, %.3f, %.3f",
                        pickInfoOnHost.normalInWorld.x,
                        pickInfoOnHost.normalInWorld.y,
                        pickInfoOnHost.normalInWorld.z);
            ImGui::Text("Albedo: %.3f, %.3f, %.3f",
                        pickInfoOnHost.albedo.x,
                        pickInfoOnHost.albedo.y,
                        pickInfoOnHost.albedo.z);
            ImGui::Text("Emittance: %.3f, %.3f, %.3f",
                        pickInfoOnHost.emittance.x,
                        pickInfoOnHost.emittance.y,
                        pickInfoOnHost.emittance.z);
            ImGui::Separator();

            ImGui::PushID("Candidates");
            ImGui::Text("#Candidates: %u", 1 << log2NumCandidateSamples); ImGui::SameLine();
            if (ImGui::Button("-"))
                log2NumCandidateSamples = std::max(log2NumCandidateSamples - 1, 0);
            ImGui::SameLine();
            if (ImGui::Button("+"))
                log2NumCandidateSamples = std::min(log2NumCandidateSamples + 1, 8);
            ImGui::PopID();

            ImGui::PushID("Samples");
            ImGui::Text("#Samples: %u", 1 << log2NumSamples); ImGui::SameLine();
            if (ImGui::Button("-"))
                log2NumSamples = std::max(log2NumSamples - 1, 0);
            ImGui::SameLine();
            if (ImGui::Button("+"))
                log2NumSamples = std::min({ log2NumSamples + 1, log2NumCandidateSamples, 3 });
            ImGui::PopID();

            if (ImGui::Checkbox("Temporal Denoiser", &useTemporalDenosier)) {
                CUDADRV_CHECK(cuStreamSynchronize(cuStream));
                denoiser.destroy();

                OptixDenoiserModelKind modelKind = useTemporalDenosier ?
                    OPTIX_DENOISER_MODEL_KIND_TEMPORAL :
                    OPTIX_DENOISER_MODEL_KIND_HDR;
                denoiser = gpuEnv.optixContext.createDenoiser(modelKind, true, true);

                size_t stateSize;
                size_t scratchSize;
                size_t scratchSizeForComputeIntensity;
                uint32_t numTasks;
                denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                                 &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                                 &numTasks);
                hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
                hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
                hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
                denoiserStateBuffer.resize(stateSize, 1);
                denoiserScratchBuffer.resize(std::max(scratchSize, scratchSizeForComputeIntensity), 1);

                denoisingTasks.resize(numTasks);
                denoiser.getTasks(denoisingTasks.data());

                denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);
            }

            ImGui::Text("Buffer to Display");
            ImGui::RadioButtonE("Noisy Beauty", &bufferTypeToDisplay, Shared::BufferToDisplay::NoisyBeauty);
            ImGui::RadioButtonE("Albedo", &bufferTypeToDisplay, Shared::BufferToDisplay::Albedo);
            ImGui::RadioButtonE("Normal", &bufferTypeToDisplay, Shared::BufferToDisplay::Normal);
            ImGui::RadioButtonE("Flow", &bufferTypeToDisplay, Shared::BufferToDisplay::Flow);
            ImGui::RadioButtonE("Denoised Beauty", &bufferTypeToDisplay, Shared::BufferToDisplay::DenoisedBeauty);

            ImGui::SliderFloat("Motion Vector Scale", &motionVectorScale, -2.0f, 2.0f);

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float cudaFrameTime = frameIndex >= 2 ? curGPUTimer.frame.report() : 0.0f;
            float updateTime = frameIndex >= 2 ? curGPUTimer.update.report() : 0.0f;
            float renderTime = frameIndex >= 2 ? curGPUTimer.render.report() : 0.0f;
            float denoiseTime = frameIndex >= 2 ? curGPUTimer.denoise.report() : 0.0f;
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime);
            ImGui::Text("  Update: %.3f [ms]", updateTime);
            ImGui::Text("  Render: %.3f [ms]", renderTime);
            ImGui::Text("  Denoise: %.3f [ms]", denoiseTime);

            ImGui::End();
        }



        curGPUTimer.frame.start(cuStream);

        //// JP: 各インスタンスのトランスフォームを更新する。
        //// EN: Update the transform of each instance.
        //if (animate || lastFrameWasAnimated) {
        //    for (int i = 0; i < bunnyInsts.size(); ++i) {
        //        MovingInstance &bunnyInst = bunnyInsts[i];
        //        bunnyInst.update(animate ? 1.0f / 60.0f : 0.0f);
        //        // TODO: まとめて送る。
        //        CUDADRV_CHECK(cuMemcpyHtoDAsync(instDataBuffer.getCUdeviceptrAt(bunnyInst.ID),
        //                                        &bunnyInst.instData, sizeof(bunnyInsts[i].instData), cuStream));
        //    }
        //}

        // JP: IASのリビルドを行う。
        //     アップデートの代用としてのリビルドでは、インスタンスの追加・削除や
        //     ASビルド設定の変更を行っていないのでmarkDirty()やprepareForBuild()は必要無い。
        // EN: Rebuild the IAS.
        //     Rebuild as the alternative for update doesn't involves
        //     add/remove of instances and changes of AS build settings
        //     so neither of markDirty() nor prepareForBuild() is required.
        curGPUTimer.update.start(cuStream);
        if (animate)
            plp.travHandle = ias.rebuild(cuStream, iasInstanceBuffer, iasMem, asScratchMem);
        curGPUTimer.update.stop(cuStream);

        // Render
        bool reservoirSizeChanged = log2NumSamples != plp.log2NumSamples;
        bool firstAccumFrame = reservoirSizeChanged || animate || cameraIsActuallyMoving || resized || frameIndex == 0;
        bool resetFlowBuffer = reservoirSizeChanged || resized || frameIndex == 0;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;
        else
            ++numAccumFrames;
        plp.numAccumFrames = numAccumFrames;
        plp.log2NumCandidateSamples = log2NumCandidateSamples;
        if (reservoirSizeChanged) {
            plp.log2NumSamples = log2NumSamples;
            if (log2NumSamples == 0)
                plp.reservoirBuffer.n1 = reservoirBuffer_n1.getBlockBuffer2D();
            else if (log2NumSamples == 1)
                plp.reservoirBuffer.n2 = reservoirBuffer_n2.getBlockBuffer2D();
            else if (log2NumSamples == 2)
                plp.reservoirBuffer.n4 = reservoirBuffer_n4.getBlockBuffer2D();
            else if (log2NumSamples == 3)
                plp.reservoirBuffer.n8 = reservoirBuffer_n8.getBlockBuffer2D();
        }
        plp.instanceDataBuffer = gpuEnv.instDataBuffer[bufferIndex].getDevicePointer();
        plp.pickInfo = pickInfos[bufferIndex].getDevicePointer();
        plp.mousePosition = int2(static_cast<int32_t>(g_mouseX),
                                 static_cast<int32_t>(g_mouseY));
        plp.resetFlowBuffer = resetFlowBuffer;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));

        curGPUTimer.render.start(cuStream);
        gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.primaryRayGenProgram);
        gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.shadingRayGenProgram);
        gpuEnv.pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.render.stop(cuStream);

        gpuEnv.pipeline.setRayGenerationProgram(gpuEnv.pickRayGenProgram);
        gpuEnv.pipeline.launch(cuStream, plpOnDevice, 1, 1, 1);

        curGPUTimer.denoise.start(cuStream);

        // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
        // EN: Copy the results to the linear buffers (and normalize normals).
        cudau::dim3 dimCopyBuffers = kernelCopyToLinearBuffers.calcGridDim(renderTargetSizeX, renderTargetSizeY);
        kernelCopyToLinearBuffers(cuStream, dimCopyBuffers,
                          beautyAccumBuffer.getSurfaceObject(0),
                          albedoAccumBuffer.getSurfaceObject(0),
                          normalAccumBuffer.getSurfaceObject(0),
                          linearBeautyBuffer.getDevicePointer(),
                          linearAlbedoBuffer.getDevicePointer(),
                          linearNormalBuffer.getDevicePointer(),
                          uint2(renderTargetSizeX, renderTargetSizeY));

        // JP: パストレーシング結果のデノイズ。
        //     毎フレーム呼ぶ必要があるのはcomputeIntensity()とinvoke()。
        //     computeIntensity()は自作することもできる。
        //     サイズが足りていればcomputeIntensity()のスクラッチバッファーとしてデノイザーのものが再利用できる。
        // EN: Denoise the path tracing result.
        //     computeIntensity() and invoke() should be calld every frame.
        //     You can also create a custom computeIntensity().
        //     Reusing the scratch buffer for denoising for computeIntensity() is possible if its size is enough.
        denoiser.computeIntensity(cuStream,
                                  linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                                  denoiserScratchBuffer, hdrIntensity);
        //float hdrIntensityOnHost;
        //CUDADRV_CHECK(cuMemcpyDtoH(&hdrIntensityOnHost, hdrIntensity, sizeof(hdrIntensityOnHost)));
        //printf("%g\n", hdrIntensityOnHost);
        for (int i = 0; i < denoisingTasks.size(); ++i)
            denoiser.invoke(cuStream,
                            false, hdrIntensity, 0.0f,
                            linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                            linearAlbedoBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                            linearNormalBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
                            linearFlowBuffer, OPTIX_PIXEL_FORMAT_FLOAT2,
                            resetFlowBuffer ? linearBeautyBuffer : linearDenoisedBeautyBuffer,
                            linearDenoisedBeautyBuffer,
                            denoisingTasks[i]);

        outputBufferSurfaceHolder.beginCUDAAccess(cuStream);

        // JP: デノイズ結果や中間バッファーの可視化。
        // EN: Visualize the denosed result or intermediate buffers.
        void* bufferToDisplay = nullptr;
        switch (bufferTypeToDisplay) {
        case Shared::BufferToDisplay::NoisyBeauty:
            bufferToDisplay = linearBeautyBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Albedo:
            bufferToDisplay = linearAlbedoBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Normal:
            bufferToDisplay = linearNormalBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Flow:
            bufferToDisplay = linearFlowBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::DenoisedBeauty:
            bufferToDisplay = linearDenoisedBeautyBuffer.getDevicePointer();
            break;
        default:
            Assert_ShouldNotBeCalled();
            break;
        }
        kernelVisualizeToOutputBuffer(cuStream, kernelVisualizeToOutputBuffer.calcGridDim(renderTargetSizeX, renderTargetSizeY),
                                 bufferToDisplay,
                                 bufferTypeToDisplay,
                                 0.5f, std::pow(10.0f, motionVectorScale),
                                 outputBufferSurfaceHolder.getNext(),
                                 uint2(renderTargetSizeX, renderTargetSizeY));

        outputBufferSurfaceHolder.endCUDAAccess(cuStream);

        curGPUTimer.denoise.stop(cuStream);

        curGPUTimer.frame.stop(cuStream);

        if (takeScreenShot && frameIndex + 1 == 60) {
            CUDADRV_CHECK(cuStreamSynchronize(cuStream));
            auto rawImage = new float4[renderTargetSizeX * renderTargetSizeY];
            glGetTextureSubImage(
                outputTexture.getHandle(), 0,
                0, 0, 0, renderTargetSizeX, renderTargetSizeY, 1,
                GL_RGBA, GL_FLOAT, sizeof(float4) * renderTargetSizeX * renderTargetSizeY, rawImage);
            saveImage("output.png", renderTargetSizeX, renderTargetSizeY, rawImage,
                      false, true);
            delete[] rawImage;
            break;
        }



        // ----------------------------------------------------------------
        // JP: OptiXによる描画結果を表示用レンダーターゲットにコピーする。
        // EN: Copy the OptiX rendering results to the display render target.

        if (bufferTypeToDisplay == Shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == Shared::BufferToDisplay::DenoisedBeauty) {
            glEnable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyleWithGamma;
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyle;
        }

        glViewport(0, 0, curFBWidth, curFBHeight);

        glUseProgram(drawOptiXResultShader.getHandle());

        glUniform2ui(0, curFBWidth, curFBHeight);

        glBindTextureUnit(0, outputTexture.getHandle());
        glBindSampler(0, outputSampler.getHandle());

        glBindVertexArray(vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        // END: Copy the OptiX rendering results to the display render target.
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);

        ++frameIndex;
    }

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));
    gpuTimers[1].finalize();
    gpuTimers[0].finalize();



    CUDADRV_CHECK(cuMemFree(plpOnDevice));

    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();


    
    CUDADRV_CHECK(cuMemFree(hdrIntensity));

    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));
    
    denoiserScratchBuffer.finalize();
    denoiserStateBuffer.finalize();
    
    denoiser.destroy();
    
    rngBuffer.finalize();

    linearDenoisedBeautyBuffer.finalize();
    linearFlowBuffer.finalize();
    linearNormalBuffer.finalize();
    linearAlbedoBuffer.finalize();
    linearBeautyBuffer.finalize();

    normalAccumBuffer.finalize();
    albedoAccumBuffer.finalize();
    beautyAccumBuffer.finalize();



    hitGroupSBT.finalize();

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    
    gpuEnv.finalize();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}