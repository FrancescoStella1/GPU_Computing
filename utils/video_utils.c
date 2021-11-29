/*
 This file is based on examples and guidelines retrieved from https://ffmpeg.org/doxygen/2.0 and http://dranger.com/ffmpeg/tutorial01.html
*/
#include "video_utils.h"
#include "timing.c"
#include <dirent.h>
#include <sys/stat.h>

#define MAX_FRAMES   10
#define VID_CPU_TIMING   "video_timing_cpu.txt"
#define VID_GPU_TIMING   "video_timing_gpu.txt"


void saveFrame(AVFrame *frame, int width, int height, int iFrame) {
    FILE *pFile;
    char szFilename[128];
    int y;

    DIR *dir = opendir("./images/results/frames");
    if(ENOENT == errno) {
        printf("\nCreating output directory..\n");
        mkdir("./images/results/frames", 0700);
        closedir(dir);
    }

    sprintf(szFilename, "./images/results/frames/frame%d.ppm", iFrame);
    pFile = fopen(szFilename, "wb");
    if(pFile == NULL)
        return;
    
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    for(y=0; y<height; y++) {
        fwrite(frame->data[0] + y*frame->linesize[0], sizeof(uint8_t), width*3, pFile);
    }
    
    fclose(pFile);
}


void extract_frames(char *filename) {  
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

    //av_dump_format(pFormatContext, 0, filename, 0);

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
    while(av_read_frame(pFormatContext, &pkt) >= 0 && i<MAX_FRAMES) {
        // Check if packet comes from video stream and decode frame
        if(pkt.stream_index == videoStream) {
            avcodec_decode_video2(pCodecCtx, frame, &frameFinished, &pkt);
            // If we got a frame, save it
            if(frameFinished) {
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, pCodecCtx->height, frameRGB->data, frameRGB->linesize);
                saveFrame(frameRGB, pCodecCtx->width, pCodecCtx->height, i);
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
}


void process_frames(char *path, int cpu, int num_streams, int write) {
    DIR *d;
    struct dirent *dir;
    char *out_dir = "./images/results";
    char *out = (char *)calloc(strlen(out_dir)+24, sizeof(char));
    char *frames_dir = "./images/results/frames";
    char *frame = (char *)calloc(strlen(frames_dir)+14, sizeof(char));
    unsigned char *img;
    int width, height, channels;
    int frame_num = 0;

    d = opendir(path);
    if (d) {
        while((dir = readdir(d)) != NULL) {
            const char *end = strchr(dir->d_name, '.');
            if((end == NULL) || strcmp(end+1, "ppm")!=0)
                continue;
            
            char *begin = strchr(dir->d_name, 'e') + 1;
            end = strchr(begin, '.');
            if((begin == NULL) || (end == NULL))
                continue;

            begin[(int)(end - begin)] = '\0';
            frame_num = atoi(begin);
            
            sprintf(frame, "%s/%s.ppm", frames_dir, dir->d_name);
            printf("Frame path: %s\n", frame);
            img = stbi_load(frame, &width, &height, &channels, 0);
            size_t size = width * height * sizeof(unsigned char);

            // Host memory allocation and copy of the loaded image
            unsigned char *h_img = (unsigned char *)malloc(size*channels);     // 3 channels
            unsigned char *h_img_gray = (unsigned char *)malloc(size);
            memcpy(h_img, img, size*3);

            // Grayscale conversion on CPU/GPU
            if(cpu) {
                clock_t clk_start = clock();
                convert(h_img, h_img_gray, size);
                clock_t clk_end = clock();
                double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
                printf("[Grayscale conversion CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
                write_to_file(VID_CPU_TIMING, "Grayscale", clk_elapsed, 0, 0);
            }
            else {
                cuda_convert(h_img, h_img_gray, width, height, num_streams, VID_GPU_TIMING);
            }

            free(h_img);

            if(write) {
                sprintf(out, "%s/%s%d.jpg", out_dir, "grayscale_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, h_img_gray, 100);
            }

            // Gamma correction on CPU/GPU
            if(cpu) {
                struct Histogram *hist = createHistogram();
                clock_t clk_start = clock();
                gamma_correction(hist, h_img_gray, size);
                clock_t clk_end = clock();
                double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
                free(hist);
                printf("[Gamma correction CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
                write_to_file(VID_CPU_TIMING, "Gamma correction", clk_elapsed, 0, 0);
            }
            else {
                cuda_gamma_correction(h_img_gray, size, VID_GPU_TIMING);
            }

            if(write) {
                sprintf(out, "%s/%s%d.jpg", out_dir, "gamma_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, h_img_gray, 100);
            }

            unsigned char* gradientX = (unsigned char*)calloc(width*height, sizeof(unsigned char));
            unsigned char* gradientY = (unsigned char*)calloc(width*height, sizeof(unsigned char));

            // Gradients computation on CPU/GPU
            if(cpu) {
                clock_t clk_start = clock();
                convolutionHorizontal(h_img_gray, gradientX, height, width);
                convolutionVertical(h_img_gray, gradientY, height, width);
                clock_t clk_end = clock();
                double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
                printf("[Gradients computation CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
                write_to_file(VID_CPU_TIMING, "Gradients", clk_elapsed, 0, 0);
            }
            else {
                cuda_compute_gradients(h_img_gray, gradientX, gradientY, width, height, VID_GPU_TIMING);
            }

            free(h_img_gray);

            if(write) {
                sprintf(out, "%s/%s%d.jpg", out_dir, "gradientX_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, gradientX, 100);
                sprintf(out, "%s/%s%d.jpg", out_dir, "gradientY_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, gradientY, 100);
            }

            unsigned char *magnitude = (unsigned char *)calloc(width*height, sizeof(unsigned char));
            unsigned char *direction = (unsigned char *)calloc(width*height, sizeof(unsigned char));

            // Magnitude and Direction computation on CPU/GPU
            if(cpu) {
                clock_t clk_start = clock();
                compute_magnitude(gradientX, gradientY, magnitude, width*height);
                compute_direction(gradientX, gradientY, direction, width*height);
                clock_t clk_end = clock();
                double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
                printf("[Magnitude & Direction CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
                write_to_file(VID_CPU_TIMING, "Magnitude and Direction", clk_elapsed, 0, 0);
            }
            else {
                cuda_compute_mag_dir(gradientX, gradientY, magnitude, direction, width*height, VID_GPU_TIMING);
            }

            free(gradientX);
            free(gradientY);

            if(write) {
                sprintf(out, "%s/%s%d.jpg", out_dir, "magnitude_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, magnitude, 100);
                sprintf(out, "%s/%s%d.jpg", out_dir, "direction_frame", frame_num);
                stbi_write_jpg(out, width, height, 1, direction, 100);
            }

            // HOG computation on CPU/GPU
            float *hog;

            if(cpu) {
                clock_t clk_start = clock();
                compute_hog(hog, magnitude, direction, width, height);
                clock_t clk_end = clock();
                double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
                printf("[HOG computation CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
                write_to_file(VID_CPU_TIMING, "HOG computation", clk_elapsed, 0, 1);
            }
            else {
                cuda_compute_hog(hog, magnitude, direction, width, height, VID_GPU_TIMING);
            }
            frame_num++;
            free(hog);
        }
        closedir(d);
    }
    printf("\n\n[ DONE ]\n\n");
}