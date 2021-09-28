#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>


static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL, *audio_dec_ctx;
static AVStream *video_stream = NULL, *audio_stream = NULL;
static const char *src_filename = NULL;
static const char *video_dst_filename = NULL;
static const char *audio_dst_filename = NULL;
static FILE *video_dst_file = NULL;
static FILE *audio_dst_file = NULL;

static uint8_t *video_dst_data[4] = {NULL};
static int video_dst_linesize[4];
static int video_dst_bufsize;

static uint8_t **audio_dst_data = NULL;
static int audio_dst_linesize;
static int audio_dst_bufsize;

static int video_stream_idx = -1, audio_stream_idx = -1;
static AVFrame *frame = NULL;
static AVPacket pkt;
static int video_frame_count = 0;
static int audio_frame_count = 0;


static int decode_packet(int *got_frame, int cached) {
    int ret = 0;

    if(pkt.stream_index == video_stream_idx) {
        // Decode video frame
        //ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);           // Deprecated
        if(video_dec_ctx->codec_type == AVMEDIA_TYPE_VIDEO || video_dec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
            ret = avcodec_send_packet(video_dec_ctx, &pkt);
            if(ret < 0) { //} && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                fprintf(stderr, "Error decoding video frame\n");
                return ret;
            }
            else {
                if(ret >= 0)
                    pkt.size = 0;
                ret = avcodec_receive_frame(video_dec_ctx, frame);
                if(ret >= 0)
                    *got_frame = 1;
            }
        }
 
        if(*got_frame) {
            //printf("Video_frame%s n: %d coded_n: %d pts:%s\n", cached ? "(cached)" : "", video_frame_count++,
                //frame->coded_picture_number, av_ts2timestr(frame->pts, &video_dec_ctx->time_base));
            /* Copy decoded frame to destination buffer: this is required since rawvideo expects non aligned data */
            av_image_copy(video_dst_data, video_dst_linesize, (const uint8_t **)(frame->data), frame->linesize, video_dec_ctx->pix_fmt,
                video_dec_ctx->width, video_dec_ctx->height);
            // Write to rawvideo file
            fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
        }
    }
    return 0;
}


void process_video(const char *filename) {

}