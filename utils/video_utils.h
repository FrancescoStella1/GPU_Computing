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


/**
 * @brief Saves a frame in a .ppm file.
 * 
 * @param frame pointer to the decoded video frame.
 * @param width width of the picture to save.
 * @param height height of the picture to save.
 * @param iFrame index of the frame.
*/
void save_frame(AVFrame *frame, int width, int height, int iFrame);


/**
 * @brief Extracts frames from a video.
 * 
 * @param filename path to the video file.
 * @param num_frames maximum number of frames to extract.
*/
void extract_frames(char *filename, int num_frames);


/**
 * @brief Applies the HOG algorithm to all the frames in a videoclip.
 * 
 * @param filepath path to the video file.
 * @param cpu integer indicating if CPU is used (1: CPU, 0: GPU).
 * @param num_streams number of non-null streams to use.
 * @param write_images integer indicating if images must be stored.
 * @param write_timing integer indicating if processing time must be stored.
*/
void process_frames(char *filepath, int cpu, int num_streams, int write_images, int write_timing);