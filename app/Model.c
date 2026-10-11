#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <errno.h>
#include <jpeglib.h>

#include "larod.h"
#include "ACAP.h"
#include "Model.h"
#include "imgutils.h"
#include "labelparse.h"
#include "model_params.h"
#include "model_quant.h"  // Generated at build time by extract_model_params.py

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)   { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)   {}
#define MODEL_MAX_CACHED_CROPS 5

static bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd);
float iou(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2);
void Model_Cleanup();
cJSON* non_maximum_suppression(cJSON* list);
static void clear_crop_cache(void);
// YOLOv8 exports produce two uint8 output tensors, cut before the final
// concat so each gets its own quantization scale (see tools/export_yolov8.sh
// -onimc): coords [1,4,boxes] in model-input pixels, scores [1,nc,boxes]
// already sigmoid'd, with no objectness channel.
static int coordOutIdx = 0;
static int scoreOutIdx = 1;
static void* larodOutput2Addr = MAP_FAILED;
static int larodOutput2Fd = -1;
static size_t output2BufferSize = 0;
static size_t coordRowStride = 0;
static size_t scoreRowStride = 0;
static float coordQuant = 1.0, coordZero = 0;
static float scoreQuant = 1.0, scoreZero = 0;
static char OBJECT_DETECTOR_OUT2_FILE_PATTERN[] = "/tmp/larod.out2.test-XXXXXX";

// Model and video dimensions
static unsigned int modelWidth = 640;
static unsigned int modelHeight = 640;
static unsigned int videoWidth = 856;  // Default for 640 model: 4:3 aspect (856x640)
static unsigned int videoHeight = 640;
static unsigned int channels = 3;
static unsigned int boxes = 0;
static unsigned int classes = 0;
static size_t inputs = 1;
static size_t outputs = 1;
static size_t ppInputs = 1;
static size_t ppOutputs = 1;
static float confidenceThreshold = 0.30;  // from settings "confidence" (percent)
static float nms = 0.45;
static int larodModelFd = -1;
static larodConnection* conn = NULL;
static larodModel* InfModel = NULL;
static larodModel* ppModel = NULL;
static larodJobRequest* ppReq = NULL;
static larodMap* ppMap;
static larodJobRequest* infReq;
static void* ppInputAddr = MAP_FAILED;
static void* larodInputAddr = MAP_FAILED;
static void* larodOutput1Addr = MAP_FAILED;
static int ppInputFd = -1;
static int larodInputFd = -1;
static int larodOutput1Fd = -1;
static larodTensor** inputTensors = NULL;
static larodTensor** outputTensors = NULL;
static larodTensor** ppInputTensors = NULL;
static larodTensor** ppOutputTensors = 0;
static size_t yuyvBufferSize = 0;
static size_t outputBufferSize = 0;
//For cropping
static unsigned char* original_rgb_buffer = NULL;
larodMap* ppMapHD               = NULL;
larodModel* ppModelHD           = NULL;
larodTensor** ppInputTensorsHD  = NULL;
size_t ppNumInputsHD            = 1;
larodTensor** ppOutputTensorsHD = NULL;
size_t ppNumOutputsHD           = 1;
larodJobRequest* ppReqHD        = NULL;
void* ppInputAddrHD             = MAP_FAILED;
void* ppOutputAddrHD            = MAP_FAILED;
int ppInputFdHD                 = -1;
int ppOutputFdHD                = -1;

static cJSON* modelConfig = 0;

// Runtime label parsing
static char** modelLabels = NULL;
static char* labelBuffer = NULL;
static size_t numLabels = 0;
static const char* DEFAULT_MODEL_PATH = "model/model.tflite";
static const char* DEFAULT_LABELS_PATH = "model/labels.txt";
static const char* CUSTOM_MODEL_PATH = "localdata/model.tflite";
static const char* CUSTOM_LABELS_PATH = "localdata/labels.txt";
static const char* activeModelPath = NULL;
static const char* activeLabelsPath = NULL;

static const char* CLASSIFIER_MODEL_PATH = "model/classifier.tflite";
static const char* CLASSIFIER_LABELS_PATH = "model/classifier-labels.txt";
static larodModel* classifierModel = NULL;
static larodJobRequest* classifierReq = NULL;
static larodTensor** classifierInputTensors = NULL;
static larodTensor** classifierOutputTensors = NULL;
static void* classifierInputAddr = MAP_FAILED;
static void* classifierOutputAddr = MAP_FAILED;
static int classifierInputFd = -1;
static int classifierOutputFd = -1;
static int classifierModelFd = -1;
static size_t classifierInputSize = 0;
static size_t classifierOutputSize = 0;
static unsigned int classifierWidth = 0;
static unsigned int classifierHeight = 0;
static unsigned int classifierChannels = 0;
static unsigned int classifierClasses = 0;
static float classifierQuant = 1.0f;
static int classifierZero = 0;
static float classifierCropMargin = 1.25f;
static int classifierHighRes = 1;   // settings "classifierHighRes"
static char** classifierLabels = NULL;
static char* classifierLabelBuffer = NULL;
static size_t numClassifierLabels = 0;
static int classifierReady = 0;
static char CLASSIFIER_INPUT_FILE_PATTERN[] = "/tmp/larod.classifier.in-XXXXXX";
static char CLASSIFIER_OUTPUT_FILE_PATTERN[] = "/tmp/larod.classifier.out-XXXXXX";

static char PP_SD_INPUT_FILE_PATTERN[] = "/tmp/larod.pp.test-XXXXXX";
static char OBJECT_DETECTOR_INPUT_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
static char OBJECT_DETECTOR_OUT1_FILE_PATTERN[]  = "/tmp/larod.out1.test-XXXXXX";
char PP_HD_INPUT_FILE_PATTERN[]  = "/tmp/larod.pp.hd.test-XXXXXX";
char PP_HD_OUTPUT_FILE_PATTERN[] = "/tmp/larod.pp.hd.out.test-XXXXXX";

int inferenceErrors = 5;
static int currentRefId = 1;

typedef struct {
    int refId;
    int crop_x;
    int crop_y;
    int crop_w;
    int crop_h;
	int img_w;
	int img_h;	
    unsigned char* jpeg_buf;
    unsigned jpeg_size;
} CropCacheEntry;

static CropCacheEntry cropCache[MODEL_MAX_CACHED_CROPS];
static int numCropCache = 0;

static void clear_crop_cache(void) {
    for (int i = 0; i < numCropCache; i++) {
        if (cropCache[i].jpeg_buf) {
            free(cropCache[i].jpeg_buf);
            cropCache[i].jpeg_buf = NULL;
        }
    }
    numCropCache = 0;
}


// larod runs a warmup inference when a model is loaded, and that warmup fails
// if the DLPU is not ready to serve us yet -- another app still holding it, a
// previous instance of this one not fully torn down, or the device still
// settling after a boot. The failure is transient ("Could not run warmup job:
// Failure when invoking interpreter"), so back off and try again rather than
// leaving the app dead until someone restarts it by hand. Axis's own reference
// ACAP retries model loading for the same reason.
#define MODEL_LOAD_ATTEMPTS 5

static larodModel*
load_model_with_retry(const larodDevice* device) {
    larodError* error = NULL;
    larodModel* loaded = NULL;

    for (int attempt = 1; attempt <= MODEL_LOAD_ATTEMPTS; attempt++) {
        loaded = larodLoadModel(conn, larodModelFd, device, LAROD_ACCESS_PRIVATE,
                                "object_detection", NULL, &error);
        if (loaded) {
            if (attempt > 1)
                LOG("Model loaded on attempt %d\n", attempt);
            return loaded;
        }

        LOG_WARN("%s: Model load attempt %d/%d failed: %s\n", __func__, attempt,
                 MODEL_LOAD_ATTEMPTS, (error && error->msg) ? error->msg : "unknown");
        larodClearError(&error);

        if (attempt == MODEL_LOAD_ATTEMPTS)
            break;

        // larod read from the fd on the failed attempt, so rewind before retrying.
        if (lseek(larodModelFd, 0, SEEK_SET) == (off_t)-1) {
            LOG_WARN("%s: Could not rewind model file: %s\n", __func__, strerror(errno));
            break;
        }
        useconds_t backoff = 1000000u << (attempt - 1);  // 1s, 2s, 4s, 8s
        LOG("Retrying model load in %u ms\n", backoff / 1000);
        usleep(backoff);
    }

    return NULL;
}

// HD preprocessing produces a full-resolution RGB frame for detection crops.
// It costs videoWidth*videoHeight*4.5 bytes of mapped buffers (about 4 MB at
// 720p) plus a larod model and job, so it is built on first use rather than at
// startup -- most installs never turn cropping on. Failure disables crops
// instead of taking the whole model down.
static int hdReady = 0;
static int hdFailed = 0;

static int
hd_preprocessing_setup(void) {
    larodError* error = NULL;
    size_t hdInputs = 1;
    size_t hdOutputs = 1;

    if (hdReady)
        return 1;
    if (hdFailed)
        return 0;

    LOG("Setting up HD preprocessing for detection crops (%ux%u)\n", videoWidth, videoHeight);

    ppMapHD = larodCreateMap(&error);
    if (!ppMapHD ||
        !larodMapSetStr(ppMapHD, "image.input.format", "nv12", &error) ||
        !larodMapSetIntArr2(ppMapHD, "image.input.size", videoWidth, videoHeight, &error) ||
        !larodMapSetStr(ppMapHD, "image.output.format", "rgb-interleaved", &error) ||
        !larodMapSetIntArr2(ppMapHD, "image.output.size", videoWidth, videoHeight, &error)) {
        LOG_WARN("%s: Failed building HD preprocessing map: %s\n",
                 __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }

    const larodDevice* device = larodGetDevice(conn, "cpu-proc", 0, &error);
    if (!device) {
        LOG_WARN("%s: Could not get cpu-proc device: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }

    ppModelHD = larodLoadModel(conn, -1, device, LAROD_ACCESS_PRIVATE, "", ppMapHD, &error);
    if (!ppModelHD) {
        LOG_WARN("%s: Unable to load HD preprocessing model: %s\n",
                 __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }

    ppInputTensorsHD = larodCreateModelInputs(ppModelHD, &hdInputs, &error);
    ppOutputTensorsHD = larodCreateModelOutputs(ppModelHD, &hdOutputs, &error);
    if (!ppInputTensorsHD || !ppOutputTensorsHD) {
        LOG_WARN("%s: Failed retrieving HD tensors: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }
    ppNumInputsHD = hdInputs;
    ppNumOutputsHD = hdOutputs;

    if (!createAndMapTmpFile(PP_HD_INPUT_FILE_PATTERN, videoWidth * videoHeight * 3 / 2,
                             &ppInputAddrHD, &ppInputFdHD) ||
        !createAndMapTmpFile(PP_HD_OUTPUT_FILE_PATTERN, videoWidth * videoHeight * 3,
                             &ppOutputAddrHD, &ppOutputFdHD)) {
        LOG_WARN("%s: Could not allocate HD preprocessing buffers\n", __func__);
        hdFailed = 1;
        return 0;
    }

    if (!larodSetTensorFd(ppInputTensorsHD[0], ppInputFdHD, &error) ||
        !larodSetTensorFd(ppOutputTensorsHD[0], ppOutputFdHD, &error)) {
        LOG_WARN("%s: Failed setting HD tensor fds: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }

    ppReqHD = larodCreateJobRequest(ppModelHD, ppInputTensorsHD, ppNumInputsHD,
                                    ppOutputTensorsHD, ppNumOutputsHD, NULL, &error);
    if (!ppReqHD) {
        LOG_WARN("%s: Failed creating HD preprocessing job: %s\n",
                 __func__, error ? error->msg : "");
        larodClearError(&error);
        hdFailed = 1;
        return 0;
    }

    hdReady = 1;
    return 1;
}

static int
classifier_setup(const larodDevice* device) {
    larodError* error = NULL;
    size_t inputCount = 1;
    size_t outputCount = 1;

    if (!ACAP_FILE_Exists(CLASSIFIER_MODEL_PATH)) {
        LOG("Stage 2 model not present; running detector-only\n");
        return 0;
    }
    classifierModelFd = open(CLASSIFIER_MODEL_PATH, O_RDONLY);
    if (classifierModelFd < 0) {
        LOG_WARN("%s: Could not open %s: %s\n", __func__, CLASSIFIER_MODEL_PATH, strerror(errno));
        return 0;
    }
    classifierModel = larodLoadModel(conn, classifierModelFd, device, LAROD_ACCESS_PRIVATE,
                                     "classifier", NULL, &error);
    if (!classifierModel) {
        LOG_WARN("%s: Unable to load stage 2: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        return 0;
    }
    classifierInputTensors = larodCreateModelInputs(classifierModel, &inputCount, &error);
    classifierOutputTensors = larodCreateModelOutputs(classifierModel, &outputCount, &error);
    if (!classifierInputTensors || !classifierOutputTensors || inputCount != 1 || outputCount != 1) {
        LOG_WARN("%s: Stage 2 requires one input and one output tensor\n", __func__);
        larodClearError(&error);
        return 0;
    }

    const larodTensorDims* inputDims = larodGetTensorDims(classifierInputTensors[0], &error);
    const larodTensorDims* outputDims = larodGetTensorDims(classifierOutputTensors[0], &error);
    if (!inputDims || !outputDims) {
        LOG_WARN("%s: Failed to inspect stage 2 tensors: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        return 0;
    }
    classifierHeight = inputDims->dims[1];
    classifierWidth = inputDims->dims[2];
    classifierChannels = inputDims->dims[3];
    classifierClasses = outputDims->dims[1];
    if (classifierWidth == 0 || classifierHeight == 0 || classifierChannels != 3 || classifierClasses == 0) {
        LOG_WARN("%s: Stage 2 must have NHWC RGB input and class-vector output\n", __func__);
        return 0;
    }

    if (!labels_parse_file(CLASSIFIER_LABELS_PATH, &classifierLabels, &classifierLabelBuffer,
                           &numClassifierLabels) || numClassifierLabels != classifierClasses) {
        LOG_WARN("%s: %s must contain %u classifier labels\n", __func__,
                 CLASSIFIER_LABELS_PATH, classifierClasses);
        return 0;
    }

    const larodTensorPitches* inputPitches = larodGetTensorPitches(classifierInputTensors[0], &error);
    const larodTensorPitches* outputPitches = larodGetTensorPitches(classifierOutputTensors[0], &error);
    if (!inputPitches || !outputPitches) {
        LOG_WARN("%s: Failed to inspect stage 2 tensor pitches: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        return 0;
    }
    classifierInputSize = inputPitches->pitches[0];
    classifierOutputSize = outputPitches->pitches[0];
    if (!createAndMapTmpFile(CLASSIFIER_INPUT_FILE_PATTERN, classifierInputSize,
                             &classifierInputAddr, &classifierInputFd) ||
        !createAndMapTmpFile(CLASSIFIER_OUTPUT_FILE_PATTERN, classifierOutputSize,
                             &classifierOutputAddr, &classifierOutputFd)) {
        LOG_WARN("%s: Could not allocate stage 2 buffers\n", __func__);
        return 0;
    }
    if (!larodSetTensorFd(classifierInputTensors[0], classifierInputFd, &error) ||
        !larodSetTensorFd(classifierOutputTensors[0], classifierOutputFd, &error)) {
        LOG_WARN("%s: Failed setting stage 2 tensor fds: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        return 0;
    }
    classifierReq = larodCreateJobRequest(classifierModel, classifierInputTensors, 1,
                                           classifierOutputTensors, 1, NULL, &error);
    if (!classifierReq) {
        LOG_WARN("%s: Failed creating stage 2 job: %s\n", __func__, error ? error->msg : "");
        larodClearError(&error);
        return 0;
    }
    if (!tflite_first_output_quant(CLASSIFIER_MODEL_PATH, &classifierQuant, &classifierZero)) {
        classifierQuant = 1.0f / 255.0f;
        classifierZero = 0.0f;
        LOG_WARN("%s: Could not read stage 2 quantization; using uint8 defaults\n", __func__);
    }
    classifierReady = 1;
    LOG("Stage 2 ready: %u classes at %ux%u\n", classifierClasses, classifierWidth, classifierHeight);
    return 1;
}

// High-resolution stage 2 source. main.c hands over the newest frame of a
// second NV12 stream at the sensor's largest 16:9 size before each inference.
// The detector input is the same scene squashed to modelWidth x modelHeight
// without letterboxing, so normalized boxes map onto it directly.
static const uint8_t* hiresData = NULL;
static unsigned int hiresWidth = 0;
static unsigned int hiresHeight = 0;
static unsigned int hiresPitch = 0;
static unsigned int lastCropMicros = 0;

// NV12 -> RGB. Stage 1 and the stage 2 calibration crops went through larod's
// cpu-proc conversion, whose matrix is not documented, so the hi-res path picks
// whichever of these best reproduces the detector's RGB buffer (see
// hires_detect_matrix) instead of guessing.
typedef struct {
    const char* name;
    float yOffset, yGain, rV, gU, gV, bU;
} YuvMatrix;
static const YuvMatrix yuvMatrices[] = {
    {"bt601-limited", 16.0f, 1.164f, 1.596f, -0.391f, -0.813f, 2.018f},
    {"bt601-full",     0.0f, 1.000f, 1.402f, -0.344f, -0.714f, 1.772f},
    {"bt709-limited", 16.0f, 1.164f, 1.793f, -0.213f, -0.533f, 2.112f},
    {"bt709-full",     0.0f, 1.000f, 1.575f, -0.187f, -0.468f, 1.856f},
};
#define NUM_YUV_MATRICES (sizeof(yuvMatrices) / sizeof(yuvMatrices[0]))
static int hiresMatrix = -1;   // index into yuvMatrices, -1 until detected

static inline uint8_t
clamp_u8(float v) {
    return (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
}

static inline void
yuv_to_rgb(const YuvMatrix* m, float y, float u, float v, float out[3]) {
    float luma = (y - m->yOffset) * m->yGain;
    u -= 128.0f;
    v -= 128.0f;
    out[0] = luma + m->rV * v;
    out[1] = luma + m->gU * u + m->gV * v;
    out[2] = luma + m->bU * u;
}

// Mean Y/U/V over a block of the hi-res frame, clamped to the frame.
static void
hires_block_yuv(int x0, int y0, int x1, int y1, float* y, float* u, float* v) {
    const uint8_t* uvPlane = hiresData + (size_t)hiresPitch * hiresHeight;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int)hiresWidth) x1 = hiresWidth;
    if (y1 > (int)hiresHeight) y1 = hiresHeight;
    unsigned long sy = 0, su = 0, sv = 0, n = 0;
    for (int yy = y0; yy < y1; yy++) {
        const uint8_t* row = hiresData + (size_t)yy * hiresPitch;
        const uint8_t* uvRow = uvPlane + (size_t)(yy >> 1) * hiresPitch;
        for (int xx = x0; xx < x1; xx++) {
            sy += row[xx];
            su += uvRow[xx & ~1];
            sv += uvRow[(xx & ~1) + 1];
            n++;
        }
    }
    if (!n) n = 1;
    *y = (float)sy / n;
    *u = (float)su / n;
    *v = (float)sv / n;
}

// Compare a grid of detector-input pixels against the matching hi-res blocks
// under each candidate matrix and keep the closest. Runs once per model setup;
// the two streams are captured back to back, so a static scene dominates.
static void
hires_detect_matrix(void) {
    if (hiresMatrix >= 0 || !hiresData || larodInputAddr == MAP_FAILED)
        return;
    const uint8_t* rgb = (const uint8_t*)larodInputAddr;
    const float sx = (float)hiresWidth / modelWidth;
    const float sy = (float)hiresHeight / modelHeight;
    double error[NUM_YUV_MATRICES] = {0};
    unsigned long samples = 0;
    for (unsigned int gy = 0; gy < 18; gy++) {
        for (unsigned int gx = 0; gx < 32; gx++) {
            unsigned int mx = (gx * 2 + 1) * modelWidth / 64;
            unsigned int my = (gy * 2 + 1) * modelHeight / 36;
            float y, u, v;
            hires_block_yuv((int)(mx * sx), (int)(my * sy),
                            (int)((mx + 1) * sx + 0.999f), (int)((my + 1) * sy + 0.999f),
                            &y, &u, &v);
            const uint8_t* ref = rgb + ((size_t)my * modelWidth + mx) * 3;
            for (unsigned int m = 0; m < NUM_YUV_MATRICES; m++) {
                float out[3];
                yuv_to_rgb(&yuvMatrices[m], y, u, v, out);
                for (int c = 0; c < 3; c++)
                    error[m] += fabsf((float)clamp_u8(out[c]) - ref[c]);
            }
            samples++;
        }
    }
    unsigned int best = 0;
    for (unsigned int m = 1; m < NUM_YUV_MATRICES; m++)
        if (error[m] < error[best])
            best = m;
    hiresMatrix = (int)best;
    LOG("Stage 2 hi-res colour: %s (mean abs error/channel %.1f; %s %.1f, %s %.1f, %s %.1f, %s %.1f)\n",
        yuvMatrices[best].name, error[best] / (samples * 3),
        yuvMatrices[0].name, error[0] / (samples * 3), yuvMatrices[1].name, error[1] / (samples * 3),
        yuvMatrices[2].name, error[2] / (samples * 3), yuvMatrices[3].name, error[3] / (samples * 3));
}

// Same square crop as classifier_fill_input (centre, longest side x margin,
// out-of-frame black) but cut from the hi-res NV12 frame. When the crop is
// larger than the classifier input each output pixel averages a k x k grid of
// source samples (area resampling, k <= 4) so far hands do not alias; when it
// is smaller it falls back to bilinear.
static void
classifier_fill_input_hires(float nx, float ny, float nw, float nh) {
    const YuvMatrix* matrix = &yuvMatrices[hiresMatrix >= 0 ? hiresMatrix : 0];
    const uint8_t* uvPlane = hiresData + (size_t)hiresPitch * hiresHeight;
    uint8_t* destination = (uint8_t*)classifierInputAddr;
    float centerX = (nx + nw * 0.5f) * hiresWidth;
    float centerY = (ny + nh * 0.5f) * hiresHeight;
    float side = fmaxf(nw * hiresWidth, nh * hiresHeight) * classifierCropMargin;
    if (side < 2.0f) side = 2.0f;
    float left = centerX - side * 0.5f;
    float top = centerY - side * 0.5f;
    float step = side / classifierWidth;
    int k = (int)ceilf(step);
    if (k < 1) k = 1;
    if (k > 4) k = 4;
    const int maxX = (int)hiresWidth - 1;
    const int maxY = (int)hiresHeight - 1;

    for (unsigned int outputY = 0; outputY < classifierHeight; outputY++) {
        for (unsigned int outputX = 0; outputX < classifierWidth; outputX++) {
            float sumY = 0.0f, sumU = 0.0f, sumV = 0.0f;
            int inside = 0;
            for (int j = 0; j < k; j++) {
                float sourceY = top + (outputY + (j + 0.5f) / k) * step - 0.5f;
                if (sourceY < -0.5f || sourceY > maxY + 0.5f)
                    continue;
                for (int i = 0; i < k; i++) {
                    float sourceX = left + (outputX + (i + 0.5f) / k) * step - 0.5f;
                    if (sourceX < -0.5f || sourceX > maxX + 0.5f)
                        continue;
                    if (k > 1) {
                        // Box filter: nearest sample per sub-cell.
                        int x = (int)(sourceX + 0.5f), y = (int)(sourceY + 0.5f);
                        if (x > maxX) x = maxX;
                        if (y > maxY) y = maxY;
                        const uint8_t* uv = uvPlane + (size_t)(y >> 1) * hiresPitch + (x & ~1);
                        sumY += hiresData[(size_t)y * hiresPitch + x];
                        sumU += uv[0];
                        sumV += uv[1];
                    } else {
                        int x1 = (int)floorf(sourceX), y1 = (int)floorf(sourceY);
                        float fx = sourceX - x1, fy = sourceY - y1;
                        int x0 = x1 < 0 ? 0 : x1, y0 = y1 < 0 ? 0 : y1;
                        int x2 = x1 + 1 > maxX ? maxX : x1 + 1, y2 = y1 + 1 > maxY ? maxY : y1 + 1;
                        const uint8_t* r0 = hiresData + (size_t)y0 * hiresPitch;
                        const uint8_t* r1 = hiresData + (size_t)y2 * hiresPitch;
                        sumY += (r0[x0] * (1.0f - fx) + r0[x2] * fx) * (1.0f - fy) +
                                (r1[x0] * (1.0f - fx) + r1[x2] * fx) * fy;
                        // Chroma is subsampled 2x; nearest is ample at this scale.
                        int cx = (int)(sourceX + 0.5f), cy = (int)(sourceY + 0.5f);
                        if (cx > maxX) cx = maxX;
                        if (cy > maxY) cy = maxY;
                        if (cx < 0) cx = 0;
                        if (cy < 0) cy = 0;
                        const uint8_t* uv = uvPlane + (size_t)(cy >> 1) * hiresPitch + (cx & ~1);
                        sumU += uv[0];
                        sumV += uv[1];
                    }
                    inside++;
                }
            }
            uint8_t* output = destination + (outputY * classifierWidth + outputX) * 3;
            if (!inside) {
                output[0] = output[1] = output[2] = 0;
                continue;
            }
            // The conversion is affine, so converting the in-frame mean and
            // scaling by the in-frame fraction equals averaging RGB with
            // out-of-frame samples as black.
            float rgb[3];
            yuv_to_rgb(matrix, sumY / inside, sumU / inside, sumV / inside, rgb);
            float coverage = (float)inside / (k * k);
            for (int c = 0; c < 3; c++)
                output[c] = clamp_u8(rgb[c] * coverage);
        }
    }
}

// Full-resolution RGB copy of a region of the hi-res frame, for detection
// export. Caller frees.
static unsigned char*
hires_crop_rgb(int cropX, int cropY, int cropW, int cropH) {
    hires_detect_matrix();
    const YuvMatrix* matrix = &yuvMatrices[hiresMatrix >= 0 ? hiresMatrix : 0];
    const uint8_t* uvPlane = hiresData + (size_t)hiresPitch * hiresHeight;
    unsigned char* rgb = malloc((size_t)cropW * cropH * 3);
    if (!rgb)
        return NULL;
    for (int y = 0; y < cropH; y++) {
        const uint8_t* row = hiresData + (size_t)(cropY + y) * hiresPitch;
        const uint8_t* uvRow = uvPlane + (size_t)((cropY + y) >> 1) * hiresPitch;
        unsigned char* output = rgb + (size_t)y * cropW * 3;
        for (int x = 0; x < cropW; x++) {
            int sx = cropX + x;
            float out[3];
            yuv_to_rgb(matrix, row[sx], uvRow[sx & ~1], uvRow[(sx & ~1) + 1], out);
            output[x * 3 + 0] = clamp_u8(out[0]);
            output[x * 3 + 1] = clamp_u8(out[1]);
            output[x * 3 + 2] = clamp_u8(out[2]);
        }
    }
    return rgb;
}

void
Model_SetClassifierFrame(VdoBuffer* buffer, unsigned int width, unsigned int height, unsigned int pitch) {
    hiresData = NULL;
    if (!buffer || !classifierHighRes || width == 0 || height == 0 || pitch < width)
        return;
    VdoFrame* frame = vdo_buffer_get_frame(buffer);
    size_t needed = (size_t)pitch * height * 3 / 2;
    if (frame && vdo_frame_get_size(frame) > 0 && vdo_frame_get_size(frame) < needed) {
        LOG_WARN("%s: Hi-res frame is %zu bytes, NV12 %ux%u pitch %u needs %zu; using detector frame\n",
                 __func__, (size_t)vdo_frame_get_size(frame), width, height, pitch, needed);
        classifierHighRes = 0;
        return;
    }
    hiresData = (const uint8_t*)vdo_buffer_get_data(buffer);
    hiresWidth = width;
    hiresHeight = height;
    hiresPitch = pitch;
}

void
Model_GetCropFrameSize(unsigned int* width, unsigned int* height) {
    if (width) *width = hiresData ? hiresWidth : videoWidth;
    if (height) *height = hiresData ? hiresHeight : videoHeight;
}

int
Model_WantsClassifierFrame(void) {
    return classifierReady && classifierHighRes;
}

static void
classifier_fill_input(float nx, float ny, float nw, float nh) {
    const uint8_t* source = (const uint8_t*)larodInputAddr;
    uint8_t* destination = (uint8_t*)classifierInputAddr;
    float centerX = (nx + nw * 0.5f) * modelWidth;
    float centerY = (ny + nh * 0.5f) * modelHeight;
    float side = fmaxf(nw * modelWidth, nh * modelHeight) * classifierCropMargin;
    if (side < 2.0f) side = 2.0f;
    float left = centerX - side * 0.5f;
    float top = centerY - side * 0.5f;

    for (unsigned int outputY = 0; outputY < classifierHeight; outputY++) {
        float sourceY = top + (outputY + 0.5f) * side / classifierHeight - 0.5f;
        int sourceY1 = (int)floorf(sourceY);
        float yFraction = sourceY - sourceY1;
        for (unsigned int outputX = 0; outputX < classifierWidth; outputX++) {
            float sourceX = left + (outputX + 0.5f) * side / classifierWidth - 0.5f;
            int sourceX1 = (int)floorf(sourceX);
            float xFraction = sourceX - sourceX1;
            uint8_t* output = destination + (outputY * classifierWidth + outputX) * 3;
            for (unsigned int channel = 0; channel < 3; channel++) {
                float value = 0.0f;
                for (int dy = 0; dy < 2; dy++) {
                    int y = sourceY1 + dy;
                    for (int dx = 0; dx < 2; dx++) {
                        int x = sourceX1 + dx;
                        float sample = 0.0f;
                        if (x >= 0 && x < (int)modelWidth && y >= 0 && y < (int)modelHeight)
                            sample = source[(y * modelWidth + x) * 3 + channel];
                        value += sample * (dx ? xFraction : 1.0f - xFraction) *
                                 (dy ? yFraction : 1.0f - yFraction);
                    }
                }
                output[channel] = (uint8_t)fminf(255.0f, fmaxf(0.0f, value));
            }
        }
    }
}

static void
classifier_classify_list(cJSON* list) {
    if (!classifierReady || !list || larodInputAddr == MAP_FAILED)
        return;
    if (hiresData)
        hires_detect_matrix();

    // Classifier threshold (settings, read per frame so a change applies at once):
    // classifierConfidence: percent, 0 = off. Below it the classifier's best guess is not
    // trusted and the detection gets classifierFallbackLabel instead (empty = keep the
    // detector's own label). A fallback keeps the detector's confidence as "c": the
    // detection itself is still valid, only its class is uncertain, and main.c's
    // confidence filter must not drop it for the classifier's low score.
    cJSON* settings = ACAP_Get_Config("settings");
    cJSON* minItem = settings ? cJSON_GetObjectItem(settings, "classifierConfidence") : NULL;
    cJSON* fallbackItem = settings ? cJSON_GetObjectItem(settings, "classifierFallbackLabel") : NULL;
    float minConfidence = (minItem && cJSON_IsNumber(minItem)) ? (float)minItem->valuedouble / 100.0f : 0.0f;
    const char* fallbackLabel = (fallbackItem && cJSON_IsString(fallbackItem)) ? fallbackItem->valuestring : "";

    cJSON* detection = NULL;
    cJSON_ArrayForEach(detection, list) {
        cJSON* x = cJSON_GetObjectItem(detection, "x");
        cJSON* y = cJSON_GetObjectItem(detection, "y");
        cJSON* width = cJSON_GetObjectItem(detection, "w");
        cJSON* height = cJSON_GetObjectItem(detection, "h");
        cJSON* confidence = cJSON_GetObjectItem(detection, "c");
        if (!x || !y || !width || !height)
            continue;

        struct timeval cropStart, cropEnd;
        gettimeofday(&cropStart, NULL);
        if (hiresData)
            classifier_fill_input_hires((float)x->valuedouble, (float)y->valuedouble,
                                        (float)width->valuedouble, (float)height->valuedouble);
        else
            classifier_fill_input((float)x->valuedouble, (float)y->valuedouble,
                                  (float)width->valuedouble, (float)height->valuedouble);
        gettimeofday(&cropEnd, NULL);
        lastCropMicros = (unsigned int)((cropEnd.tv_sec - cropStart.tv_sec) * 1000000 +
                                        (cropEnd.tv_usec - cropStart.tv_usec));
        larodError* error = NULL;
        if (!larodRunJob(conn, classifierReq, &error)) {
            LOG_WARN("%s: Stage 2 inference failed: %s\n", __func__, error ? error->msg : "");
            larodClearError(&error);
            return;
        }
        const uint8_t* scores = (const uint8_t*)classifierOutputAddr;
        unsigned int best = 0;
        for (unsigned int index = 1; index < classifierClasses; index++) {
            if (scores[index] > scores[best])
                best = index;
        }
        float bestConfidence = ((float)scores[best] - classifierZero) * classifierQuant;
        const char* bestLabel = labels_get(classifierLabels, numClassifierLabels, best);
        // hand_c (percent, like "c" after main.c's conversion) = the detector's confidence
        // that this is a hand. When the threshold rule replaces the label, the classifier's
        // best guess is kept as gesture/gesture_c; otherwise it is simply label/c.
        if (confidence)
            cJSON_AddNumberToObject(detection, "hand_c", (int)(confidence->valuedouble * 100.0));
        if (minConfidence > 0.0f && bestConfidence < minConfidence) {
            cJSON_AddStringToObject(detection, "gesture", bestLabel);
            cJSON_AddNumberToObject(detection, "gesture_c", (int)(bestConfidence * 100.0f));
            cJSON_AddTrueToObject(detection, "fallback");
            if (fallbackLabel[0]) {
                cJSON_DeleteItemFromObject(detection, "label");
                cJSON_AddStringToObject(detection, "label", fallbackLabel);
            }
            continue;   // keep the detector's label (if no fallback label) and confidence
        }
        cJSON_DeleteItemFromObject(detection, "label");
        cJSON_AddStringToObject(detection, "label", bestLabel);
        cJSON_DeleteItemFromObject(detection, "c");
        cJSON_AddNumberToObject(detection, "c", bestConfidence);
    }
}

cJSON*
Model_Inference(VdoBuffer* image) {
    larodError* error = NULL;
    LOG_TRACE("%s: Called\n", __func__);
    if (!image) {
        LOG_TRACE("%s: No image\n", __func__);
        return 0;
    }
    if (ACAP_STATUS_Bool("model", "state") == 0) {
        LOG_TRACE("%s: Model not running\n", __func__);
        return 0;
    }
    if (inferenceErrors <= 0) {
        LOG_WARN("Too many inference errors.  Model stopped\n");
        Model_Cleanup();
        return 0;
    }

    // Get the captured NV12 frame
    uint8_t* nv12Data = (uint8_t*)vdo_buffer_get_data(image);
    
    // Copy NV12 data to BOTH preprocessing input buffers
    memcpy(ppInputAddr, nv12Data, yuyvBufferSize);      // For model inference (Aspect 1:1)


    // Cropping output config
    cJSON* settings = ACAP_Get_Config("settings");
    if (!settings) {
		LOG_TRACE("ERROR %s>\n",__func__);
		return 0;
	}
    cJSON* cropping = cJSON_GetObjectItem(settings, "cropping");
    cJSON* sdCapture = cJSON_GetObjectItem(settings, "sd_capture");
    int cropping_active = cropping && cJSON_IsTrue(cJSON_GetObjectItem(cropping, "active"));
    int sdcard_active   = (cropping && cJSON_IsTrue(cJSON_GetObjectItem(cropping, "sdcard"))) ||
                          (sdCapture && cJSON_IsTrue(cJSON_GetObjectItem(sdCapture, "enabled")));
    int fullframe_needed = cropping_active || sdcard_active;
    // When VDO already delivers model-sized frames the model input RGB is the full frame.
    int capture_is_model_size = (videoWidth == modelWidth && videoHeight == modelHeight);
	original_rgb_buffer = 0;
	if( fullframe_needed && !capture_is_model_size && hd_preprocessing_setup() ) {
		memcpy(ppInputAddrHD, nv12Data, yuyvBufferSize);    // For HD preprocessing (original res)

		if (lseek(ppOutputFdHD, 0, SEEK_SET) == -1) {
			LOG_WARN("%s: Unable to rewind HD output file position: %s\n", __func__, strerror(errno));
			inferenceErrors--;
			return 0;
		}

		// Run HD preprocessing job
		if (!larodRunJob(conn, ppReqHD, &error)) {
			LOG_WARN("%s: Unable to run HD pre-processing job: %s (%d)\n", __func__, error->msg, error->code);
			ACAP_STATUS_SetString("output", "cropError", "HD preprocessing failed - crops unavailable");
			ACAP_STATUS_SetNumber("output", "cropErrorTime", ACAP_DEVICE_Timestamp());
			larodClearError(&error);
			inferenceErrors--;
			return 0;
		}
		original_rgb_buffer = (unsigned char*)ppOutputAddrHD;
	}

    // Run standard preprocessing for model inference
    if (!larodRunJob(conn, ppReq, &error)) {
        LOG_WARN("%s: Unable to run job to preprocess model: %s (%d)\n", __func__, error->msg, error->code);
        ACAP_STATUS_SetString("model", "status", "Preprocessing failed");
        ACAP_STATUS_SetString("model", "error", "Image preprocessing failed");
        larodClearError(&error);
        inferenceErrors--;
        return 0;
    }
    if (fullframe_needed && capture_is_model_size)
        original_rgb_buffer = (unsigned char*)larodInputAddr;
    
    if (lseek(larodOutput1Fd, 0, SEEK_SET) == -1) {
        LOG_WARN("%s: Unable to rewind output file position: %s\n", __func__, strerror(errno));
        inferenceErrors--;
        return 0;
    }
    
    // Run inference
    if (!larodRunJob(conn, infReq, &error)) {
        LOG_WARN("%s: Unable to run inference on model: %s (%d)\n", __func__, error->msg, error->code);
        ACAP_STATUS_SetString("model", "status", "Inference failed");
        ACAP_STATUS_SetString("model", "error", "Model inference execution failed");
        larodClearError(&error);
        inferenceErrors--;
        return 0;
    }

    const uint8_t* coordData = (const uint8_t*)larodOutput1Addr;
    const uint8_t* scoreData = (const uint8_t*)larodOutput2Addr;
    cJSON* list = cJSON_CreateArray();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long timestamp = tv.tv_sec * 1000LL + tv.tv_usec / 1000;

    LOG_TRACE("%s: Processing %u boxes, %u classes, confidence >= %.2f\n",
              __func__, boxes, classes, confidenceThreshold);

    // Both tensors are channel-major: channel c of box i sits at c*stride + i.
    const uint8_t* cxRow = coordData + 0 * coordRowStride;
    const uint8_t* cyRow = coordData + 1 * coordRowStride;
    const uint8_t* cwRow = coordData + 2 * coordRowStride;
    const uint8_t* chRow = coordData + 3 * coordRowStride;

    // Coordinates dequantize to model-input pixels; fold the normalization to
    // 0..1 into the scale so the inner loop is one multiply. main.c takes it
    // from 0..1 to the 0..1000 space everything else uses.
    const float xScale = coordQuant / (float)modelWidth;
    const float yScale = coordQuant / (float)modelHeight;

    // Thresholding happens in raw uint8 space so the per-class scan stays
    // integer: this runs boxes*classes times per frame (403k for COCO).
    int rawThreshold = (int)(confidenceThreshold / scoreQuant + scoreZero);
    if (rawThreshold < 0) rawThreshold = 0;

    int detections = 0;

    for (unsigned int i = 0; i < boxes; i++) {
        unsigned int bestRaw = 0;
        int classId = -1;
        for (unsigned int c = 0; c < classes; c++) {
            unsigned int v = scoreData[c * scoreRowStride + i];
            if (v > bestRaw) {
                bestRaw = v;
                classId = c;
            }
        }
        if (classId < 0 || (int)bestRaw < rawThreshold)
            continue;

        float maxConfidence = ((float)bestRaw - scoreZero) * scoreQuant;
        float x = ((float)cxRow[i] - coordZero) * xScale;
        float y = ((float)cyRow[i] - coordZero) * yScale;
        float w = ((float)cwRow[i] - coordZero) * xScale;
        float h = ((float)chRow[i] - coordZero) * yScale;

        detections++;
        cJSON* detection = cJSON_CreateObject();
        const char* label = labels_get(modelLabels, numLabels, classId);
        cJSON_AddStringToObject(detection, "label", label);
        cJSON_AddNumberToObject(detection, "c", maxConfidence);
        // Stored as top-left corner; the model emits centre coordinates.
        cJSON_AddNumberToObject(detection, "x", x - (w / 2));
        cJSON_AddNumberToObject(detection, "y", y - (h / 2));
        cJSON_AddNumberToObject(detection, "w", w);
        cJSON_AddNumberToObject(detection, "h", h);
        cJSON_AddNumberToObject(detection, "timestamp", timestamp);
        cJSON_AddNumberToObject(detection, "refId", currentRefId++);
        cJSON_AddItemToArray(list, detection);

        if (detections <= 3)
            LOG_TRACE("%s: Detection %d: %s conf=%.2f x=%.3f y=%.3f w=%.3f h=%.3f\n",
                      __func__, detections, label, maxConfidence, x, y, w, h);
    }

    LOG_TRACE("%s: %d detections above threshold\n", __func__, detections);

    // Clear any previous errors on successful inference
    ACAP_STATUS_SetNull("model", "error");
    ACAP_STATUS_SetString("model", "status", "Running");

    cJSON* finalList = non_maximum_suppression(list);
    classifier_classify_list(finalList);
    return finalList;
}

//The detection coordinates are in pixels relative to model input dimensions
const unsigned char*
Model_GetImageData(const cJSON* detection, unsigned* jpeg_size, int* out_x, int* out_y, int* out_w, int* out_h, int* img_w, int* img_h ) {
    if (jpeg_size) *jpeg_size = 0;
    if (!detection) {
        LOG_WARN("%s: detection is NULL\n", __func__);
        return NULL;
    }
    LOG_TRACE("<%s\n", __func__);

    char* json = cJSON_PrintUnformatted(detection);
    if (json) {
        LOG_TRACE("%s", json);
        free(json);
    }

    cJSON* settings = ACAP_Get_Config("settings");
    cJSON* cropping = settings ? cJSON_GetObjectItem(settings, "cropping") : NULL;
    int cropping_active = cropping && cJSON_IsTrue(cJSON_GetObjectItem(cropping, "active"));
    if (!cropping_active)
        return NULL;

    cJSON* refIdObj = cJSON_GetObjectItem(detection, "refId");
    if (!refIdObj || !cJSON_IsNumber(refIdObj)) {
        LOG_WARN("%s: detection missing valid 'refId'\n", __func__);
        return NULL;
    }
    int refId = refIdObj->valueint;

    for (int i = 0; i < numCropCache; ++i) {
        if (cropCache[i].refId == refId) {
            if (jpeg_size) *jpeg_size = cropCache[i].jpeg_size;
            if (out_x) *out_x = cropCache[i].crop_x;
            if (out_y) *out_y = cropCache[i].crop_y;
            if (out_w) *out_w = cropCache[i].crop_w;
            if (out_h) *out_h = cropCache[i].crop_h;
            if (img_w) *img_w = cropCache[i].img_w;
            if (img_h) *img_h = cropCache[i].img_h;
            return cropCache[i].jpeg_buf;
        }
    }

    int leftborder_px = cropping && cJSON_GetObjectItem(cropping, "leftborder") ? cJSON_GetObjectItem(cropping, "leftborder")->valueint : 0;
    int rightborder_px = cropping && cJSON_GetObjectItem(cropping, "rightborder") ? cJSON_GetObjectItem(cropping, "rightborder")->valueint : 0;
    int topborder_px = cropping && cJSON_GetObjectItem(cropping, "topborder") ? cJSON_GetObjectItem(cropping, "topborder")->valueint : 0;
    int bottomborder_px = cropping && cJSON_GetObjectItem(cropping, "bottomborder") ? cJSON_GetObjectItem(cropping, "bottomborder")->valueint : 0;

    cJSON* xObj = cJSON_GetObjectItem(detection, "x");
    cJSON* yObj = cJSON_GetObjectItem(detection, "y");
    cJSON* wObj = cJSON_GetObjectItem(detection, "w");
    cJSON* hObj = cJSON_GetObjectItem(detection, "h");
    if (!xObj || !cJSON_IsNumber(xObj) ||
        !yObj || !cJSON_IsNumber(yObj) ||
        !wObj || !cJSON_IsNumber(wObj) ||
        !hObj || !cJSON_IsNumber(hObj)) {
        LOG_WARN("%s: detection missing geometry\n", __func__);
        return NULL;
    }

	// Coordinates are in the 0..1000 normalized space, scale to frame pixels.
	// With a hi-res stage 2 stream the export crops from that frame too, so
	// exported crops (and their borders, in those pixels) carry full sensor
	// detail and match what the classifier saw.
	int useHires = hiresData != NULL;
	int frameWidth = useHires ? (int)hiresWidth : (int)videoWidth;
	int frameHeight = useHires ? (int)hiresHeight : (int)videoHeight;
	double scale_x = (double)frameWidth / 1000.0;
	double scale_y = (double)frameHeight / 1000.0;
	int det_pixel_x = (int)round(xObj->valuedouble * scale_x);
	int det_pixel_y = (int)round(yObj->valuedouble * scale_y);
	int det_pixel_w = (int)round(wObj->valuedouble * scale_x);
	int det_pixel_h = (int)round(hObj->valuedouble * scale_y);

    int crop_x = det_pixel_x - leftborder_px;
    int crop_y = det_pixel_y - topborder_px;
    int crop_w = det_pixel_w + leftborder_px + rightborder_px;
    int crop_h = det_pixel_h + topborder_px + bottomborder_px;

    if (crop_x < 0) { crop_w += crop_x; crop_x = 0; }
    if (crop_y < 0) { crop_h += crop_y; crop_y = 0; }
    if (crop_x + crop_w > frameWidth) crop_w = frameWidth - crop_x;
    if (crop_y + crop_h > frameHeight) crop_h = frameHeight - crop_y;
    if (crop_w < 1) crop_w = 1;
    if (crop_h < 1) crop_h = 1;

    int det_x = det_pixel_x - crop_x;
    int det_y = det_pixel_y - crop_y;
    int det_w = det_pixel_w;
    int det_h = det_pixel_h;
    if (det_x < 0) { det_w += det_x; det_x = 0; }
    if (det_y < 0) { det_h += det_y; det_y = 0; }
    if (det_x + det_w > crop_w) det_w = crop_w - det_x;
    if (det_y + det_h > crop_h) det_h = crop_h - det_y;
    if (det_w < 1) det_w = 1;
    if (det_h < 1) det_h = 1;

    if (crop_x >= frameWidth || crop_y >= frameHeight)
        return NULL;

    if (!useHires && !original_rgb_buffer) {
        LOG_WARN("%s: Original RGB image buffer is NULL\n", __func__);
        ACAP_STATUS_SetString("output", "cropError", "Image buffer unavailable - cropping may be disabled");
        ACAP_STATUS_SetNumber("output", "cropErrorTime", ACAP_DEVICE_Timestamp());
        return NULL;
    }


    unsigned char* crop_buf = useHires ?
        hires_crop_rgb(crop_x, crop_y, crop_w, crop_h) :
        crop_interleaved(original_rgb_buffer, videoWidth, videoHeight, 3, crop_x, crop_y, crop_w, crop_h);
    if (!crop_buf) {
        LOG_WARN("%s: failed to crop interleaved RGB buffer\n", __func__);
        return NULL;
    }

    unsigned char* jpeg_buf = NULL;
    unsigned long jpeglen = 0;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    cinfo.image_width = crop_w;
    cinfo.image_height = crop_h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);

    buffer_to_jpeg(crop_buf, &cinfo, &jpeglen, &jpeg_buf);

    jpeg_destroy_compress(&cinfo);
    free(crop_buf);

    if (!jpeg_buf || jpeglen == 0) {
        LOG_WARN("%s: JPEG encoding failed\n", __func__);
        ACAP_STATUS_SetString("output", "cropError", "JPEG encoding failed");
        ACAP_STATUS_SetNumber("output", "cropErrorTime", ACAP_DEVICE_Timestamp());
        return NULL;
    }

    // Clear crop error on success
    ACAP_STATUS_SetNull("output", "cropError");

    if (numCropCache < MODEL_MAX_CACHED_CROPS) {
        cropCache[numCropCache].refId = refId;
        cropCache[numCropCache].crop_x = det_x;
        cropCache[numCropCache].crop_y = det_y;
        cropCache[numCropCache].crop_w = det_w;
        cropCache[numCropCache].crop_h = det_h;
        cropCache[numCropCache].img_w = crop_w;
        cropCache[numCropCache].img_h = crop_h;
        cropCache[numCropCache].jpeg_buf = jpeg_buf;
        cropCache[numCropCache].jpeg_size = jpeglen;
        numCropCache++;
    }

    if (jpeg_size) *jpeg_size = (unsigned)jpeglen;
    if (out_x) *out_x = det_x;
    if (out_y) *out_y = det_y;
    if (out_w) *out_w = det_w;
    if (out_h) *out_h = det_h;
	*img_w = crop_w;
	*img_h = crop_h;

    LOG_TRACE("%s>\n", __func__);
    return jpeg_buf;
}

void Model_Reset(void) {
    clear_crop_cache();
}

// Raw output tensor bytes, for off-camera comparison against a CPU run.
// which=0 gives the coord tensor, which=1 the score tensor.
const void* Model_GetRawOutput(int which, size_t* size) {
    if (size) *size = 0;
    if (which == 0 && larodOutput1Addr != MAP_FAILED) {
        if (size) *size = outputBufferSize;
        return larodOutput1Addr;
    }
    if (which == 1 && larodOutput2Addr != MAP_FAILED) {
        if (size) *size = output2BufferSize;
        return larodOutput2Addr;
    }
    return NULL;
}

// Raw statistics for the buffers on both sides of the inference, published
// over MQTT for debugging. A constant input buffer means preprocessing never
// wrote it; a score tensor pinned near 255 means the DLPU is producing
// saturated nonsense rather than the model being wrong.
cJSON* Model_GetDebugStats(void) {
    cJSON* stats = cJSON_CreateObject();
    if (!stats)
        return NULL;

    cJSON_AddNumberToObject(stats, "modelWidth", modelWidth);
    cJSON_AddNumberToObject(stats, "modelHeight", modelHeight);
    cJSON_AddNumberToObject(stats, "videoWidth", videoWidth);
    cJSON_AddNumberToObject(stats, "videoHeight", videoHeight);
    cJSON_AddNumberToObject(stats, "boxes", boxes);
    cJSON_AddNumberToObject(stats, "classes", classes);
    if (classifierReady) {
        cJSON_AddStringToObject(stats, "classifierSource", hiresData ? "hires" : "detector");
        cJSON_AddNumberToObject(stats, "hiresWidth", hiresWidth);
        cJSON_AddNumberToObject(stats, "hiresHeight", hiresHeight);
        cJSON_AddStringToObject(stats, "hiresColour", hiresMatrix >= 0 ? yuvMatrices[hiresMatrix].name : "pending");
        cJSON_AddNumberToObject(stats, "classifierCropUs", lastCropMicros);
    }

    if (larodInputAddr != MAP_FAILED) {
        const uint8_t* in = (const uint8_t*)larodInputAddr;
        size_t n = (size_t)modelWidth * modelHeight * channels;
        unsigned lo = 255, hi = 0;
        unsigned long sum = 0;
        for (size_t k = 0; k < n; k++) {
            if (in[k] < lo) lo = in[k];
            if (in[k] > hi) hi = in[k];
            sum += in[k];
        }
        cJSON_AddNumberToObject(stats, "inputMin", lo);
        cJSON_AddNumberToObject(stats, "inputMax", hi);
        cJSON_AddNumberToObject(stats, "inputMean", n ? (double)sum / n : 0);
    }

    if (larodOutput1Addr != MAP_FAILED) {
        const uint8_t* c = (const uint8_t*)larodOutput1Addr;
        unsigned lo = 255, hi = 0;
        for (size_t k = 0; k < outputBufferSize; k++) {
            if (c[k] < lo) lo = c[k];
            if (c[k] > hi) hi = c[k];
        }
        cJSON_AddNumberToObject(stats, "coordMin", lo);
        cJSON_AddNumberToObject(stats, "coordMax", hi);
    }

    if (larodOutput2Addr != MAP_FAILED) {
        const uint8_t* q = (const uint8_t*)larodOutput2Addr;
        unsigned lo = 255, hi = 0;
        unsigned long sum = 0;
        unsigned long over = 0;
        for (size_t k = 0; k < output2BufferSize; k++) {
            if (q[k] < lo) lo = q[k];
            if (q[k] > hi) hi = q[k];
            sum += q[k];
            if (q[k] >= 128) over++;   // >= 0.5 confidence
        }
        cJSON_AddNumberToObject(stats, "scoreMin", lo);
        cJSON_AddNumberToObject(stats, "scoreMax", hi);
        cJSON_AddNumberToObject(stats, "scoreMean",
                                output2BufferSize ? (double)sum / output2BufferSize : 0);
        cJSON_AddNumberToObject(stats, "scoresOver50pct", (double)over);
    }

    return stats;
}

// The exact RGB frame handed to the model, straight out of larod's convert
// step. Serving this makes preprocessing faults visible: a sheared or
// mis-strided image, wrong colour order, or a squashed aspect all show up
// instantly here, and none of them are distinguishable from DLPU misbehaviour
// by looking at detections alone.
unsigned char* Model_GetModelInputJPEG(unsigned* jpeg_size, unsigned* width, unsigned* height) {
    if (jpeg_size) *jpeg_size = 0;
    if (width) *width = modelWidth;
    if (height) *height = modelHeight;
    if (larodInputAddr == MAP_FAILED) {
        LOG_WARN("%s: Model input buffer not mapped\n", __func__);
        return NULL;
    }

    unsigned char* jpeg_buf = NULL;
    unsigned long jpeglen = 0;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    cinfo.image_width      = modelWidth;
    cinfo.image_height     = modelHeight;
    cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);

    buffer_to_jpeg((unsigned char*)larodInputAddr, &cinfo, &jpeglen, &jpeg_buf);
    jpeg_destroy_compress(&cinfo);

    if (!jpeg_buf || jpeglen == 0) {
        LOG_WARN("%s: JPEG encoding of model input failed\n", __func__);
        return NULL;
    }

    if (jpeg_size) *jpeg_size = (unsigned)jpeglen;
    return jpeg_buf;
}

unsigned char* Model_GetFullFrameJPEG(unsigned* jpeg_size) {
    if (jpeg_size) *jpeg_size = 0;
    if (!original_rgb_buffer) {
        LOG_WARN("%s: HD frame buffer unavailable (enable SD card or crop export)\n", __func__);
        return NULL;
    }

    unsigned char* jpeg_buf = NULL;
    unsigned long jpeglen = 0;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    cinfo.image_width      = videoWidth;
    cinfo.image_height     = videoHeight;
    cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 85, TRUE);

    buffer_to_jpeg(original_rgb_buffer, &cinfo, &jpeglen, &jpeg_buf);
    jpeg_destroy_compress(&cinfo);

    if (!jpeg_buf || jpeglen == 0) {
        LOG_WARN("%s: JPEG encoding of full frame failed\n", __func__);
        return NULL;
    }

    if (jpeg_size) *jpeg_size = (unsigned)jpeglen;
    return jpeg_buf;
}

int Model_GetLabelIndex(const char* label) {
    if (!label)
        return 0;
    if (classifierReady && classifierLabels) {
        for (size_t i = 0; i < numClassifierLabels; i++) {
            if (classifierLabels[i] && strcmp(classifierLabels[i], label) == 0)
                return (int)i;
        }
    }
    if (!modelLabels || numLabels == 0)
        return 0;
    for (size_t i = 0; i < numLabels; i++) {
        if (modelLabels[i] && strcmp(modelLabels[i], label) == 0)
            return (int)i;
    }
    return 0;
}

// IoU of two boxes given as top-left corner plus size, which is how
// detections are stored.
float iou(float ax, float ay, float aw, float ah, float bx, float by, float bw, float bh) {
    float xx1 = fmax(ax, bx);
    float yy1 = fmax(ay, by);
    float xx2 = fmin(ax + aw, bx + bw);
    float yy2 = fmin(ay + ah, by + bh);

    float inter  = fmax(0, xx2 - xx1) * fmax(0, yy2 - yy1);
    float union_ = aw * ah + bw * bh - inter;

    return union_ > 0 ? inter / union_ : 0;
}

cJSON* non_maximum_suppression(cJSON* list) {
    if (!list) {
        LOG_WARN("%s: Invalid list\n", __func__);
        return 0;
    }
    int items = cJSON_GetArraySize(list);
    if (items < 2)
        return list;

    // Copy the geometry into a flat array first. Reading it back out of cJSON
    // inside the O(n^2) comparison means a string-keyed list walk per field
    // per pair, which dominates the frame once detections number in the
    // hundreds.
    struct { float x, y, w, h, c; int keep; } *box = malloc((size_t)items * sizeof(*box));
    if (!box) {
        LOG_WARN("%s: Out of memory for %d detections\n", __func__, items);
        return list;
    }

    int n = 0;
    for (cJSON* det = list->child; det && n < items; det = det->next, n++) {
        box[n].x = cJSON_GetObjectItem(det, "x")->valuedouble;
        box[n].y = cJSON_GetObjectItem(det, "y")->valuedouble;
        box[n].w = cJSON_GetObjectItem(det, "w")->valuedouble;
        box[n].h = cJSON_GetObjectItem(det, "h")->valuedouble;
        box[n].c = cJSON_GetObjectItem(det, "c")->valuedouble;
        box[n].keep = 1;
    }

    for (int i = 0; i < n; i++) {
        if (!box[i].keep)
            continue;
        for (int j = i + 1; j < n; j++) {
            if (!box[j].keep)
                continue;
            if (iou(box[i].x, box[i].y, box[i].w, box[i].h,
                    box[j].x, box[j].y, box[j].w, box[j].h) > nms) {
                if (box[i].c > box[j].c) {
                    box[j].keep = 0;
                } else {
                    box[i].keep = 0;
                    break;
                }
            }
        }
    }

    cJSON* result = cJSON_CreateArray();
    int idx = 0;
    for (cJSON* det = list->child; det && idx < n; det = det->next, idx++) {
        if (box[idx].keep)
            cJSON_AddItemToArray(result, cJSON_Duplicate(det, 1));
    }

    free(box);
    cJSON_Delete(list);
    return result;
}

static bool 
createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd) {
	LOG_TRACE("%s: %s %zu\n", __func__,fileName, fileSize);
    int fd = mkstemp(fileName);
    if (fd < 0) {
        LOG_WARN("%s: Unable to open temp file %s: %s\n", __func__, fileName, strerror(errno));
        return false;
    }

    if (ftruncate(fd, (off_t)fileSize) < 0) {
        LOG_WARN("%s: Unable to truncate temp file %s: %s\n", __func__, fileName, strerror(errno));
        close(fd);
        return false;
    }

    if (unlink(fileName)) {
        LOG_WARN("%s: Unable to unlink from temp file %s: %s\n", __func__, fileName, strerror(errno));
        close(fd);
        return false;
    }

    void* data = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        LOG_WARN("%s: Unable to mmap temp file %s: %s\n", __func__, fileName, strerror(errno));
        close(fd);
        return false;
    }

    *mappedAddr = data;
    *convFd = fd;
    return true;
}


void
Model_Cleanup() {
    // Only the model handle is released here. We count on larod service to
    // release the privately loaded model when the session is disconnected in
    // larodDisconnect().
    larodError* error = NULL;

	clear_crop_cache();

    if (classifierReq) larodDestroyJobRequest(&classifierReq);
    if (classifierInputTensors) larodDestroyTensors(conn, &classifierInputTensors, 1, &error);
    if (classifierOutputTensors) larodDestroyTensors(conn, &classifierOutputTensors, 1, &error);
    if (classifierModel) larodDestroyModel(&classifierModel);
    if (classifierInputAddr != MAP_FAILED) munmap(classifierInputAddr, classifierInputSize);
    if (classifierOutputAddr != MAP_FAILED) munmap(classifierOutputAddr, classifierOutputSize);
    if (classifierInputFd >= 0) close(classifierInputFd);
    if (classifierOutputFd >= 0) close(classifierOutputFd);
    if (classifierModelFd >= 0) close(classifierModelFd);
    classifierInputAddr = MAP_FAILED;
    classifierOutputAddr = MAP_FAILED;
    classifierInputFd = -1;
    classifierOutputFd = -1;
    classifierModelFd = -1;
    classifierInputSize = 0;
    classifierOutputSize = 0;
    classifierReady = 0;
    hiresData = NULL;
    hiresMatrix = -1;
    labels_free(classifierLabels, classifierLabelBuffer);
    classifierLabels = NULL;
    classifierLabelBuffer = NULL;
    numClassifierLabels = 0;

	// Free runtime-parsed labels
	labels_free(modelLabels, labelBuffer);
	modelLabels = NULL;
	labelBuffer = NULL;
	numLabels = 0;

	if( ppMap ) larodDestroyMap(&ppMap);
    if( ppModel ) larodDestroyModel(&ppModel);
    larodDestroyModel(&InfModel);
    // Release everything that needs a live connection BEFORE disconnecting.
    // larodDisconnect() nulls conn and tears down the session server-side; calling
    // larodDestroyTensors(conn=NULL, ...) afterwards frees tensors the session has
    // already released, which aborts the process with "double free or corruption"
    // on every app stop.
    larodDestroyJobRequest(&ppReq);
    larodDestroyJobRequest(&infReq);
    if (inputTensors)  larodDestroyTensors(conn, &inputTensors, inputs, &error);
    if (outputTensors) larodDestroyTensors(conn, &outputTensors, outputs, &error);
    if (error) larodClearError(&error);

    if (conn) larodDisconnect(&conn, NULL);
    // Reset every fd and mapping as it is released: Model_Setup can be retried
    // after a failed load, and a second cleanup closing a stale descriptor
    // would shut down whatever unrelated fd has since taken that number.
    if (larodModelFd >= 0) close(larodModelFd);
    larodModelFd = -1;
    if (larodInputAddr != MAP_FAILED) munmap(larodInputAddr, modelWidth * modelHeight * channels);
    if (larodInputFd >= 0) close(larodInputFd);
    larodInputAddr = MAP_FAILED;
    larodInputFd = -1;
    if (ppInputAddr != MAP_FAILED) munmap(ppInputAddr, modelWidth * modelHeight * channels);
    if (ppInputFd >= 0) close(ppInputFd);
    ppInputAddr = MAP_FAILED;
    ppInputFd = -1;
    if (larodOutput1Addr != MAP_FAILED) munmap(larodOutput1Addr, outputBufferSize);
    if (larodOutput1Fd >= 0) close(larodOutput1Fd);
    if (larodOutput2Addr != MAP_FAILED) munmap(larodOutput2Addr, output2BufferSize);
    if (larodOutput2Fd >= 0) close(larodOutput2Fd);
    larodOutput1Addr = MAP_FAILED;
    larodOutput2Addr = MAP_FAILED;
    larodOutput1Fd = -1;
    larodOutput2Fd = -1;

    // HD preprocessing is built on demand, so tear it down and reset the
    // latches -- otherwise a restart would reuse freed pointers.
    if (ppInputAddrHD != MAP_FAILED) munmap(ppInputAddrHD, videoWidth * videoHeight * 3 / 2);
    if (ppInputFdHD >= 0) close(ppInputFdHD);
    if (ppOutputAddrHD != MAP_FAILED) munmap(ppOutputAddrHD, videoWidth * videoHeight * 3);
    if (ppOutputFdHD >= 0) close(ppOutputFdHD);
    ppInputAddrHD = MAP_FAILED;
    ppOutputAddrHD = MAP_FAILED;
    ppInputFdHD = -1;
    ppOutputFdHD = -1;
    if (ppMapHD) larodDestroyMap(&ppMapHD);
    if (ppModelHD) larodDestroyModel(&ppModelHD);
    larodDestroyJobRequest(&ppReqHD);
    hdReady = 0;
    hdFailed = 0;
    original_rgb_buffer = NULL;

    // Job requests and tensors are released before larodDisconnect() above --
    // see the comment there. Nothing conn-dependent may be destroyed here.
    larodClearError(&error);
	ACAP_STATUS_SetString("model","status","Model stopped");
	ACAP_STATUS_SetBool("model","state", 0);	
}



cJSON* Model_Setup(void) {
    larodError* error = NULL;
    ACAP_STATUS_SetString("model", "status", "Model initialization failed. Check log file");
    ACAP_STATUS_SetBool("model", "state", 0);

    // ==== RUNTIME MODEL INTROSPECTION ====

    // Step 1: Connect to larod
    if (!larodConnect(&conn, &error)) {
        LOG_WARN("%s: Could not connect to larod: %s\n", __func__, error ? error->msg : "unknown");
        larodClearError(&error);
        return 0;
    }

    activeModelPath = ACAP_FILE_Exists(CUSTOM_MODEL_PATH) ? CUSTOM_MODEL_PATH : DEFAULT_MODEL_PATH;
    activeLabelsPath = ACAP_FILE_Exists(CUSTOM_LABELS_PATH) ? CUSTOM_LABELS_PATH : DEFAULT_LABELS_PATH;

    // Step 2: Load model to introspect
    larodModelFd = open(activeModelPath, O_RDONLY);
    if (larodModelFd < 0) {
        LOG_WARN("%s: Could not open model %s: %s\n", __func__, activeModelPath, strerror(errno));
        Model_Cleanup();
        return 0;
    }

    // Detect platform and select appropriate chip
    const char* platform = ACAP_DEVICE_Prop("platform");
    const char* chipString = "cpu-tflite";  // Default fallback

    if (platform) {
        if (strstr(platform, "Artpec-8")) {
            chipString = "axis-a8-dlpu-tflite";
            LOG("Detected ARTPEC-8 platform\n");
        } else if (strstr(platform, "Artpec-9")) {
            chipString = "a9-dlpu-tflite";
            LOG("Detected ARTPEC-9 platform\n");
        } else {
            LOG("Using CPU inference (platform: %s)\n", platform);
        }
    } else {
        LOG_WARN("Could not detect platform, using CPU fallback\n");
    }

    // Optional override: settings "chip" forces a specific larod device, e.g.
    // "cpu-tflite" to run the same model off the DLPU. Useful for telling a
    // DLPU mis-execution apart from a model or decode problem -- if the CPU
    // produces correct detections from a model the DLPU garbles, the graph is
    // fine and the DLPU compiler is at fault.
    cJSON* chipSettings = ACAP_Get_Config("settings");
    if (chipSettings) {
        cJSON* chipOverride = cJSON_GetObjectItem(chipSettings, "chip");
        if (chipOverride && cJSON_IsString(chipOverride) && strlen(chipOverride->valuestring) > 0) {
            chipString = chipOverride->valuestring;
            LOG("Chip overridden by settings: %s\n", chipString);
        }
        cJSON* margin = cJSON_GetObjectItem(chipSettings, "classifierCropMargin");
        if (margin && cJSON_IsNumber(margin) && margin->valuedouble > 0.0)
            classifierCropMargin = (float)margin->valuedouble;
        cJSON* highRes = cJSON_GetObjectItem(chipSettings, "classifierHighRes");
        classifierHighRes = !highRes || !cJSON_IsFalse(highRes);
    }

    const larodDevice* device = larodGetDevice(conn, chipString, 0, &error);
    if (!device) {
        LOG_WARN("%s: Could not get device %s: %s\n", __func__, chipString, error->msg);
        ACAP_STATUS_SetString("model", "status", "Model load failed");
        ACAP_STATUS_SetString("model", "error", "Could not access inference hardware");
        ACAP_STATUS_SetBool("model", "state", 0);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    InfModel = load_model_with_retry(device);
    if (!InfModel) {
        ACAP_STATUS_SetString("model", "status", "Model load failed");
        ACAP_STATUS_SetString("model", "error", "Unable to load model onto the inference device");
        ACAP_STATUS_SetBool("model", "state", 0);
        Model_Cleanup();
        return 0;
    }
    classifier_setup(device);

    // Step 3: Introspect model tensors
    larodTensor** tempInputTensors = larodCreateModelInputs(InfModel, &inputs, &error);
    if (!tempInputTensors) {
        LOG_WARN("%s: Failed retrieving input tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    larodTensor** tempOutputTensors = larodCreateModelOutputs(InfModel, &outputs, &error);
    if (!tempOutputTensors) {
        LOG_WARN("%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        Model_Cleanup();
        return 0;
    }

    // Get the YOLOv8 NHWC input dimensions: batch, height, width, channels.
    const larodTensorDims* inputDims = larodGetTensorDims(tempInputTensors[0], &error);
    if (!inputDims) {
        LOG_WARN("%s: Failed to get input tensor dimensions\n", __func__);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        larodDestroyTensors(conn, &tempOutputTensors, outputs, NULL);
        Model_Cleanup();
        return 0;
    }

    modelHeight = inputDims->dims[1];
    modelWidth = inputDims->dims[2];
    channels = inputDims->dims[3];

    LOG("Model input: %ux%ux%u\n", modelWidth, modelHeight, channels);

    if (outputs != 2) {
        LOG_WARN("%s: Model has %zu output tensor(s); this app requires a YOLOv8 export "
                 "cut before the final concat into two tensors (coords + scores). "
                 "See tools/export_yolov8.sh (-onimc).\n", __func__, outputs);
        ACAP_STATUS_SetString("model", "status", "Unsupported model");
        ACAP_STATUS_SetString("model", "error",
                              "Model must have 2 output tensors (YOLOv8 split export)");
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        larodDestroyTensors(conn, &tempOutputTensors, outputs, NULL);
        Model_Cleanup();
        return 0;
    }

    // Identify coords vs scores by channel count rather than trusting the
    // tensor order, which the exporter does not guarantee.
    const larodTensorDims* dimsA = larodGetTensorDims(tempOutputTensors[0], &error);
    const larodTensorDims* dimsB = larodGetTensorDims(tempOutputTensors[1], &error);
    if (!dimsA || !dimsB) {
        LOG_WARN("%s: Failed to get output tensor dimensions\n", __func__);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        larodDestroyTensors(conn, &tempOutputTensors, outputs, NULL);
        Model_Cleanup();
        return 0;
    }
    if (dimsA->dims[1] == 4) {
        coordOutIdx = 0; scoreOutIdx = 1;
        boxes = dimsA->dims[2];
        classes = dimsB->dims[1];
    } else {
        coordOutIdx = 1; scoreOutIdx = 0;
        boxes = dimsB->dims[2];
        classes = dimsA->dims[1];
    }

    // Read quantization from the ACTUAL loaded model. larod does not expose it, and
    // the model_params.h macros describe only the model built into the package -- a
    // model uploaded at runtime via /model at a different resolution has a different
    // coordinate scale, and decoding it with the build-time scale mis-places every
    // box (a 960x544 model decoded with a 640x384 package's scale lands boxes at ~2/3
    // position, offset up-left). The parser reads the truth from the loaded file; the
    // macros remain a fallback if parsing ever fails.
    {
        float cS, sS; int cZ, sZ;
        if (tflite_output_quant(activeModelPath, &cS, &cZ, &sS, &sZ,
                                4, (int)classes)) {
            coordQuant = cS; coordZero = (float)cZ;
            scoreQuant = sS; scoreZero = (float)sZ;
            LOG("Quantization from loaded model %s: coord scale=%.6g zero=%d | score scale=%.6g zero=%d\n",
                activeModelPath, coordQuant, (int)coordZero, scoreQuant, (int)scoreZero);
        } else {
            coordQuant = COORD_QUANTIZATION_SCALE; coordZero = COORD_QUANTIZATION_ZERO_POINT;
            scoreQuant = SCORE_QUANTIZATION_SCALE; scoreZero = SCORE_QUANTIZATION_ZERO_POINT;
            LOG_WARN("%s: could not read quantization from %s; using build-time macros "
                     "(correct only for the built-in model resolution)\n",
                     __func__, activeModelPath);
        }
    }

    LOG("Model output: %u boxes, %u classes (coord tensor %d, score tensor %d)\n",
        boxes, classes, coordOutIdx, scoreOutIdx);
    LOG("Quantization: coords scale=%.9f zero=%d | scores scale=%.9f zero=%d\n",
        coordQuant, (int)coordZero, scoreQuant, (int)scoreZero);

    // Clean up temporary tensors
    larodDestroyTensors(conn, &tempInputTensors, inputs, &error);
    larodDestroyTensors(conn, &tempOutputTensors, outputs, &error);

    // Step 4: Read user settings (confidence, nms)
    cJSON* settings = ACAP_Get_Config("settings");
    if (settings) {
        // Read NMS threshold
        cJSON* nmsItem = cJSON_GetObjectItem(settings, "nms");
        if (nmsItem) {
            nms = nmsItem->valuedouble;
        }

        // Confidence is stored as a percentage and is the only threshold:
        // main.c filters on the same setting, so there is one knob, not two.
        cJSON* confidenceItem = cJSON_GetObjectItem(settings, "confidence");
        if (confidenceItem)
            confidenceThreshold = confidenceItem->valuedouble / 100.0;

        LOG("Detection thresholds: confidence=%.2f, nms=%.2f\n", confidenceThreshold, nms);
    }

    // Step 5: Capture resolution. The sensor is 16:9 and larod's convert step
    // Ask VDO for exactly the model input size. VDO scales in the ISP, so the
    // libyuv preprocessing step on the CPU then only has to convert NV12 to RGB
    // -- no resize at all. Both dimensions must be divisible by 8 for VDO.
    // Any aspect squash happens in hardware instead of on the CPU; the geometry
    // the model sees is unchanged.
    if ((modelWidth % 8) == 0 && (modelHeight % 8) == 0) {
        videoWidth = modelWidth;
        videoHeight = modelHeight;
    } else if (modelWidth <= 1280 && modelHeight <= 736) {
        videoWidth = 1280;
        videoHeight = 720;
    } else {
        videoWidth = 1920;
        videoHeight = 1080;
    }

    double modelAspect = (double)modelWidth / (double)modelHeight;
    double captureAspect = (double)videoWidth / (double)videoHeight;
    LOG("Video: %ux%u (16:9) -> model %ux%u (aspect %.2f)\n",
        videoWidth, videoHeight, modelWidth, modelHeight, modelAspect);
    if (fabs(modelAspect - captureAspect) / captureAspect > 0.15) {
        LOG_WARN("Model aspect %.2f is far from the 16:9 capture %.2f -- the scene is "
                 "squashed into the model input, which costs detection accuracy. "
                 "A 16:9-shaped model (e.g. 640x384) performs markedly better.\n",
                 modelAspect, captureAspect);
    }

    // Step 6: Load labels
    if (!labels_parse_file(activeLabelsPath, &modelLabels, &labelBuffer, &numLabels)) {
        LOG_WARN("%s: Failed to load labels from %s, using defaults\n", __func__, activeLabelsPath);
        numLabels = 0;
    } else {
        LOG("Loaded %zu labels from %s\n", numLabels, activeLabelsPath);
    }

    // Step 7: Build model.json structure for frontend/API
    modelConfig = cJSON_CreateObject();
    cJSON_AddNumberToObject(modelConfig, "modelWidth", modelWidth);
    cJSON_AddNumberToObject(modelConfig, "modelHeight", modelHeight);
    cJSON_AddNumberToObject(modelConfig, "videoWidth", videoWidth);
    cJSON_AddNumberToObject(modelConfig, "videoHeight", videoHeight);

    // Capture is always 16:9
    const char* videoAspect = "16:9";
    cJSON_AddStringToObject(modelConfig, "videoAspect", videoAspect);

    cJSON_AddNumberToObject(modelConfig, "boxes", boxes);
    cJSON_AddNumberToObject(modelConfig, "classes", classes);
    cJSON_AddNumberToObject(modelConfig, "coordScale", coordQuant);
    cJSON_AddNumberToObject(modelConfig, "scoreScale", scoreQuant);
    cJSON_AddNumberToObject(modelConfig, "confidence", confidenceThreshold);
    cJSON_AddNumberToObject(modelConfig, "nms", nms);
    cJSON_AddStringToObject(modelConfig, "path", activeModelPath);
    cJSON_AddStringToObject(modelConfig, "labelsPath", activeLabelsPath);
    cJSON_AddBoolToObject(modelConfig, "customModel", strcmp(activeModelPath, CUSTOM_MODEL_PATH) == 0);
    cJSON_AddBoolToObject(modelConfig, "customLabels", strcmp(activeLabelsPath, CUSTOM_LABELS_PATH) == 0);
    cJSON_AddStringToObject(modelConfig, "chip", chipString);
    cJSON_AddBoolToObject(modelConfig, "twoTier", classifierReady);
    if (classifierReady) {
        cJSON_AddStringToObject(modelConfig, "classifierPath", CLASSIFIER_MODEL_PATH);
        cJSON_AddStringToObject(modelConfig, "classifierLabelsPath", CLASSIFIER_LABELS_PATH);
        cJSON_AddNumberToObject(modelConfig, "classifierWidth", classifierWidth);
        cJSON_AddNumberToObject(modelConfig, "classifierHeight", classifierHeight);
        cJSON_AddNumberToObject(modelConfig, "classifierClasses", classifierClasses);
        cJSON_AddBoolToObject(modelConfig, "classifierHighRes", classifierHighRes);
    }

    // Add labels array
    cJSON* setupSettings = ACAP_Get_Config("settings");
    cJSON* labelsArray = cJSON_CreateArray();
    if (classifierReady) {
        cJSON* detectorLabels = cJSON_CreateArray();
        for (size_t i = 0; i < numLabels; i++)
            cJSON_AddItemToArray(detectorLabels, cJSON_CreateString(modelLabels[i]));
        cJSON_AddItemToObject(modelConfig, "detectorLabels", detectorLabels);
        for (size_t i = 0; i < numClassifierLabels; i++)
            cJSON_AddItemToArray(labelsArray, cJSON_CreateString(classifierLabels[i]));
        // Labels a classifier fallback can produce must have events too: the configured
        // fallback label, or (no fallback label) the detector's own labels.
        cJSON* fb = setupSettings ? cJSON_GetObjectItem(setupSettings, "classifierFallbackLabel") : NULL;
        const char* fallback = (fb && cJSON_IsString(fb)) ? fb->valuestring : "";
        for (size_t i = 0; i < (fallback[0] ? 1 : numLabels); i++) {
            const char* extra = fallback[0] ? fallback : modelLabels[i];
            int present = 0;
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, labelsArray)
                if (cJSON_IsString(item) && strcmp(item->valuestring, extra) == 0) present = 1;
            if (!present)
                cJSON_AddItemToArray(labelsArray, cJSON_CreateString(extra));
        }
    } else {
        for (size_t i = 0; i < numLabels; i++)
            cJSON_AddItemToArray(labelsArray, cJSON_CreateString(modelLabels[i]));
    }
    cJSON_AddItemToObject(modelConfig, "labels", labelsArray);

    // Store in ACAP config for other components
    ACAP_Set_Config("model", modelConfig);

    LOG("Model config: %ux%u model → %ux%u video (%s), %u boxes, %u classes\n",
        modelWidth, modelHeight, videoWidth, videoHeight, videoAspect, boxes, classes);

    // ==== END RUNTIME INTROSPECTION ====

    // Preprocessing (inference, 1:1 model)
    ppMap = larodCreateMap(&error);
    if (!ppMap) {
        LOG_WARN("%s: Could not create preprocessing larodMap %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
	
    if (!larodMapSetStr(ppMap, "image.input.format", "nv12", &error)) {
        LOG_WARN("%s: Failed setting preprocessing parameters: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    if (!larodMapSetIntArr2(ppMap, "image.input.size", videoWidth, videoHeight, &error)) {
        LOG_WARN("%s: Failed setting preprocessing parameters: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    if (!larodMapSetStr(ppMap, "image.output.format", "rgb-interleaved", &error)) {
        LOG_WARN("%s: Failed setting preprocessing parameters: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    if (!larodMapSetIntArr2(ppMap, "image.output.size", modelWidth, modelHeight, &error)) {
        LOG_WARN("%s: Failed setting preprocessing parameters: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    // Pre-processing model (1:1 model image) - model already loaded during introspection
    const char* larodLibyuvPP = "cpu-proc";
    const larodDevice* device_prePros = larodGetDevice(conn, larodLibyuvPP, 0, &error);
    if (!device_prePros) {
        LOG_WARN("%s: Could not get device %s: %s\n", __func__, larodLibyuvPP, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }
    ppModel = larodLoadModel(conn, -1, device_prePros, LAROD_ACCESS_PRIVATE, "", ppMap, &error);
    if (!ppModel) {
        LOG_WARN("%s: Unable to load preprocessing model with chip %s: %s", __func__, larodLibyuvPP, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    // Create input/output tensors
    ppInputTensors = larodCreateModelInputs(ppModel, &ppInputs, &error);
    if (!ppInputTensors) {
        LOG_WARN("%s: Failed retrieving input tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }
    ppOutputTensors = larodCreateModelOutputs(ppModel, &ppOutputs, &error);
    if (!ppOutputTensors) {
        LOG_WARN("%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }
    inputTensors = larodCreateModelInputs(InfModel, &inputs, &error);
    if (!inputTensors) {
        LOG_WARN("%s: Failed retrieving input tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }
    outputTensors = larodCreateModelOutputs(InfModel, &outputs, &error);
    if (!outputTensors) {
        LOG_WARN("%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    // Determine tensor buffer sizes
    const larodTensorPitches* ppInputPitches = larodGetTensorPitches(ppInputTensors[0], &error);
    if (!ppInputPitches) {
        LOG_WARN("%s: Could not get pitches of tensor: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    yuyvBufferSize = ppInputPitches->pitches[0];
    LOG_TRACE("Buffer size: %zu\n", yuyvBufferSize);
    
    const larodTensorPitches* ppOutputPitches = larodGetTensorPitches(ppOutputTensors[0], &error);
    if (!ppOutputPitches) {
        LOG_WARN("%s: Could not get pitches of tensor: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    size_t rgbBufferSize = ppOutputPitches->pitches[0];
    size_t expectedSize = modelWidth * modelHeight * channels;
    if (expectedSize != rgbBufferSize) {
        LOG_WARN("%s: Expected video output size %zu, actual %zu\n", __func__, expectedSize, rgbBufferSize);
        Model_Cleanup();
        return 0;
    }
    
    const larodTensorPitches* coordPitches = larodGetTensorPitches(outputTensors[coordOutIdx], &error);
    const larodTensorPitches* scorePitches = larodGetTensorPitches(outputTensors[scoreOutIdx], &error);
    if (!coordPitches || !scorePitches) {
        LOG_WARN("%s: Could not get pitches of output tensors: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    outputBufferSize  = coordPitches->pitches[0];
    output2BufferSize = scorePitches->pitches[0];
    // Last pitch entry is the byte stride between rows of dims[1] -- the
    // channel stride for these channel-major uint8 tensors.
    coordRowStride = coordPitches->pitches[coordPitches->len - 1];
    scoreRowStride = scorePitches->pitches[scorePitches->len - 1];
    LOG("Coord tensor: %zu bytes, stride %zu | Score tensor: %zu bytes, stride %zu\n",
        outputBufferSize, coordRowStride, output2BufferSize, scoreRowStride);


    // Allocate space for input tensors
    if (!createAndMapTmpFile(PP_SD_INPUT_FILE_PATTERN, yuyvBufferSize, &ppInputAddr, &ppInputFd)) {
        LOG_WARN("%s: Could not allocate pre-processor tensor\n", __func__);
        Model_Cleanup();
        return 0;
    }

    if (!createAndMapTmpFile(OBJECT_DETECTOR_INPUT_FILE_PATTERN, modelWidth * modelHeight * channels, &larodInputAddr, &larodInputFd)) {
        LOG_WARN("%s: Could not allocate input tensor\n", __func__);
        Model_Cleanup();
        return 0;
    }
    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT1_FILE_PATTERN, outputBufferSize, &larodOutput1Addr, &larodOutput1Fd)) {
        LOG_WARN("%s: Could not allocate output tensor\n", __func__);
        Model_Cleanup();
        return 0;
    }
    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT2_FILE_PATTERN, output2BufferSize, &larodOutput2Addr, &larodOutput2Fd)) {
        LOG_WARN("%s: Could not allocate score output tensor\n", __func__);
        Model_Cleanup();
        return 0;
    }

    // Connect tensors to file descriptors
    if (!larodSetTensorFd(ppInputTensors[0], ppInputFd, &error)) {
        LOG_WARN("%s: Failed setting input tensor fd: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }
    if (!larodSetTensorFd(ppOutputTensors[0], larodInputFd, &error)) {
        LOG_WARN("%s: Failed setting output tensor fd: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
        LOG_WARN("%s: Failed setting input tensor fd: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return 0;
    }

    if (!larodSetTensorFd(outputTensors[coordOutIdx], larodOutput1Fd, &error)) {
        LOG_WARN("%s: Failed setting output tensor fd: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }
    if (!larodSetTensorFd(outputTensors[scoreOutIdx], larodOutput2Fd, &error)) {
        LOG_WARN("%s: Failed setting score output tensor fd: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }

    // Create job requests
    ppReq = larodCreateJobRequest(ppModel,
                                  ppInputTensors,
                                  ppInputs,
                                  ppOutputTensors,
                                  ppOutputs,
                                  NULL,
                                  &error);
    if (!ppReq) {
        LOG_WARN("%s: Failed creating preprocessing job request: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }

    infReq = larodCreateJobRequest(InfModel,
                                   inputTensors,
                                   inputs,
                                   outputTensors,
                                   outputs,
                                   NULL,
                                   &error);
    if (!infReq) {
        LOG_WARN("%s: Failed creating inference request: %s\n", __func__, error->msg);
        Model_Cleanup();
        return 0;
    }

    clear_crop_cache();

    ACAP_STATUS_SetString("model", "status", "Model OK.");
    ACAP_STATUS_SetBool("model", "state", 1);

    return modelConfig;
}
