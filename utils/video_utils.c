/*
 This file is based on examples and guidelines retrieved from https://ffmpeg.org/doxygen/2.0 and http://dranger.com/ffmpeg/tutorial01.html
*/

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

// Compatibility with new API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif


void saveFrame(AVFrame *frame, int width, int height, int iFrame) {
    FILE *pFile;
    char szFilename[128];
    int y;

    sprintf(szFilename, "../../test_frames/frame%d.ppm", iFrame);
    pFile = fopen(szFilename, "wb");
    if(pFile == NULL)
        return;
    
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    for(y=0; y<height; y++) {
        //fwrite(frame->data[0] + y*frame->linesize[0], sizeof(uint8_t), width, pFile);
        fwrite(frame->data[0] + y*frame->linesize[0], sizeof(uint8_t), width*3, pFile);
    }
    
    fclose(pFile);
}


int process_video(char *filename) {  
    AVFormatContext *pFormatContext = avformat_alloc_context();
    AVCodecContext *pCodecCtxOrig = NULL;
    AVCodecContext *pCodecCtx = NULL;
    AVCodec *pCodec = NULL;
    AVFrame *frame = NULL;
    AVFrame *frameRGB = NULL;
    AVPacket pkt;

    int videoStream = -1;
    int numBytes;
    int frameFinished = 0;
    uint8_t *buffer = NULL;
    struct SwsContext *sws_ctx = NULL;

    // Open input file and allocate format context
    if(avformat_open_input(&pFormatContext, filename, NULL, NULL) < 0) {
        fprintf(stderr, "Could not open source video file %s.\n", filename);
        exit(1);
    }
    
    // Retrieve stream information
    if(avformat_find_stream_info(pFormatContext, NULL) < 0) {
        fprintf(stderr, "Could not find stream information.\n");
        exit(1);
    }

    av_dump_format(pFormatContext, 0, filename, 0);

    for(int i=0; i < (pFormatContext->nb_streams); i++) {
        if(pFormatContext->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }
    if(videoStream == -1) {
        fprintf(stderr, "Could not find video stream.\n");
        exit(1);
    }

    pCodecCtxOrig = pFormatContext->streams[videoStream]->codec;
    pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);

    if(pCodec == NULL) {
        fprintf(stderr, "Unsupported codec.\n");
        exit(1);
    }

    pCodecCtx = avcodec_alloc_context3(pCodec);
    if(avcodec_copy_context(pCodecCtx, pCodecCtxOrig) != 0) {
        fprintf(stderr, "Could not copy codec context.\n");
        exit(1);
    }

    if(avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec.\n");
        exit(1);
    }

    frame = av_frame_alloc();
    frameRGB = av_frame_alloc();
    if(frameRGB == NULL) {
        fprintf(stderr, "Could not allocate frame.\n");
        exit(1);
    }

    // Determine required buffer size and allocate buffer
    numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
    buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
    
    // Assign appropriate parts of buffer to image planes in frame
    avpicture_fill((AVPicture *)frameRGB, buffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

    // Initialize SWS context for scaling
    sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
    
    int i = 0;
    while(av_read_frame(pFormatContext, &pkt) >= 0 && i<10) {
        // Check if packet comes from video stream and decode frame
        if(pkt.stream_index == videoStream) {
            avcodec_decode_video2(pCodecCtx, frame, &frameFinished, &pkt);
            // If we got a frame, save it
            if(frameFinished) {
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, pCodecCtx->height, frameRGB->data, frameRGB->linesize);
                saveFrame(frameRGB, pCodecCtx->width, pCodecCtx->height, i);
                printf("\nFrame %d saved!", i);
            }
            ++i;
        }
    }
    // Free memory
    av_free_packet(&pkt);
    av_free(buffer);
    av_frame_free(&frame);

    // Close the codecs
    avcodec_close(pCodecCtx);
    avcodec_close(pCodecCtxOrig);

    // Close video file
    avformat_close_input(&pFormatContext);

    return 0;
}