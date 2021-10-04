#include "./video_sdk_interface/cuviddec.h"
#include "./video_sdk_interface/nvcuvid.h"


void process_video(char *filename) {
    // Code based on Nvidia Video Codec SDK programming guide
    // Query decode capabilities
    int coded_height = 2;
    int coded_width = 2;
    CUVIDDECODECAPS decodeCaps;
    decodeCaps.eCodecType = cudaVideoCodec_HEVC;
    decodeCaps.eChromaFormat = cudaVideoChromaFormat_420;
    decodeCaps.nBitDepthMinus8 = 2;     // 10 bit
    /*
    CUresult CUDAAPI result = cuvidGetDecoderCaps(&decodeCaps);

    // Check support for content
    if(!decodeCaps.bIsSupported) {
        fprintf(stderr, "Codec not supported on this GPU\n");
        exit(-1);
    }

    // Validate content resolution on underlying hardware
    if ((coded_width > decodeCaps.nMaxWidth) || (coded_height > decodeCaps.nMaxHeight)) {
        fprintf(stderr, "Resolution not supported on this GPU");
        exit(-1);
    }

    // Macroblock count should be <= nMaxMBCount
    if((coded_width >> 4) * (coded_height>>4) > decodeCaps.nMaxMBCount) {
        fprintf(stderr, "MBCount not supported on this GPU");
        exit(-1);
    }
    */
    CUVIDDECODECREATEINFO stDecodeCreateInfo;
    stDecodeCreateInfo.CodecType = cudaVideoCodec_H264;
    stDecodeCreateInfo.ulWidth = 1280;
    stDecodeCreateInfo.ulHeight = 720;
    stDecodeCreateInfo.ulMaxWidth = 0;
    stDecodeCreateInfo.ulMaxHeight = 0;
    stDecodeCreateInfo.ChromaFormat = cudaVideoChromaFormat_420;

    memset(&stDecodeCreateInfo, 0, sizeof(CUVIDDECODECREATEINFO));

}