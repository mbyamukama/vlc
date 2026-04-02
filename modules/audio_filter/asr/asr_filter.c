/*****************************************************************************
 * asr_filter.c : ASR (Automatic Speech Recognition) audio filter for VLC
 *
 * Taps the decoded PCM stream, feeds it to whisper.cpp, and publishes
 * recognised text via a shared libvlc variable ("asr-caption-text") that
 * the companion sub-source (modules/spu/asr_subsource.c) reads each frame.
 *
 * Audio requirements imposed by whisper.cpp:
 *   - mono, 32-bit float (FL32)
 *   - 16 000 Hz sample rate  (WHISPER_SAMPLE_RATE)
 *
 * VLC will drive format negotiation so the filter chain resamples / converts
 * before we receive samples.  We accumulate samples into a ring buffer and
 * run inference on a background thread to avoid blocking the audio thread.
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdatomic.h>
#include <string.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_filter.h>
#include <vlc_plugin.h>
#include <vlc_aout.h>
#include <vlc_threads.h>

/* whisper.cpp C API */
#include "whisper.cpp/include/whisper.h"

#define CFG_PREFIX      "asr-"
#define ASR_VAR_TEXT    "asr-caption-text"   /* shared with asr_subsource */
#define ASR_VAR_PTS     "asr-caption-pts"    /* PTS of the last segment    */

/* Accumulate this many seconds of audio before running inference.
 * Whisper works best on ≥ 1 s chunks; 3 s is a good latency/accuracy trade. */
#define ASR_CHUNK_SEC   3
#define ASR_SAMPLE_RATE WHISPER_SAMPLE_RATE                  /* 16 000 Hz  */
#define ASR_CHUNK_SAMPLES (ASR_CHUNK_SEC * ASR_SAMPLE_RATE)  /* 48 000     */

/* -------------------------------------------------------------------------
 * Private state
 * ---------------------------------------------------------------------- */
typedef struct
{
    /* whisper context – created once in Open, freed in Close */
    struct whisper_context *p_ctx;

    /* Sample accumulation ring buffer (mono FL32 @ 16 kHz) */
    float       *p_buf;          /* heap-allocated, ASR_CHUNK_SAMPLES floats */
    size_t       i_buf_fill;     /* samples currently in buffer              */
    vlc_tick_t   i_buf_pts;      /* PTS of the first sample in the buffer    */

    /* Background inference thread */
    vlc_thread_t thread;
    vlc_mutex_t  lock;
    vlc_cond_t   cond;

    /* Pending chunk handed off to the worker thread */
    float       *p_pending;      /* NULL when idle                           */
    size_t       i_pending;
    vlc_tick_t   i_pending_pts;

    bool         b_quit;         /* signals the worker to exit               */
} filter_sys_t;

/* -------------------------------------------------------------------------
 * Forward declarations
 * ---------------------------------------------------------------------- */
static int      Open    ( vlc_object_t * );
static void     Close   ( filter_t * );
static block_t *Process ( filter_t *, block_t * );
static void    *InferenceThread( void * );

/* -------------------------------------------------------------------------
 * Module descriptor
 * ---------------------------------------------------------------------- */
#define MODEL_TEXT      N_("Whisper model file")
#define MODEL_LONGTEXT  N_("Path to the ggml whisper model (e.g. ggml-base.en.bin).")
#define LANG_TEXT       N_("Language")
#define LANG_LONGTEXT   N_("BCP-47 language code, or \"auto\" for auto-detect.")
#define THREADS_TEXT    N_("Inference threads")
#define THREADS_LONGTEXT N_("Number of CPU threads used by whisper.cpp.")

vlc_module_begin()
    set_shortname( N_("ASR Captions") )
    set_description( N_("Automatic speech recognition subtitles (whisper.cpp)") )
    set_subcategory( SUBCAT_AUDIO_AFILTER )
    set_capability( "audio filter", 0 )

    add_loadfile( CFG_PREFIX "model", "",
                  MODEL_TEXT, MODEL_LONGTEXT )
    add_string(   CFG_PREFIX "language", "auto",
                  LANG_TEXT, LANG_LONGTEXT )
    add_integer(  CFG_PREFIX "threads", 4,
                  THREADS_TEXT, THREADS_LONGTEXT )
        change_integer_range( 1, 16 )

    set_callback( Open )
vlc_module_end()

/* -------------------------------------------------------------------------
 * Open
 * ---------------------------------------------------------------------- */
static int Open( vlc_object_t *p_this )
{
    filter_t     *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys    = malloc( sizeof(*p_sys) );
    if( unlikely(!p_sys) )
        return VLC_ENOMEM;

    /* --- load whisper model -------------------------------------------- */
    char *psz_model = var_InheritString( p_filter, CFG_PREFIX "model" );
    if( !psz_model || psz_model[0] == '\0' )
    {
        msg_Err( p_filter, "no whisper model specified (set asr-model)" );
        free( psz_model );
        free( p_sys );
        return VLC_EGENERIC;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;   /* falls back to CPU automatically if no GPU */

    p_sys->p_ctx = whisper_init_from_file_with_params( psz_model, cparams );
    free( psz_model );
    if( !p_sys->p_ctx )
    {
        msg_Err( p_filter, "failed to load whisper model" );
        free( p_sys );
        return VLC_EGENERIC;
    }

    /* --- sample buffer -------------------------------------------------- */
    p_sys->p_buf      = malloc( ASR_CHUNK_SAMPLES * sizeof(float) );
    if( unlikely(!p_sys->p_buf) )
    {
        whisper_free( p_sys->p_ctx );
        free( p_sys );
        return VLC_ENOMEM;
    }
    p_sys->i_buf_fill  = 0;
    p_sys->i_buf_pts   = VLC_TICK_INVALID;
    p_sys->p_pending   = NULL;
    p_sys->i_pending   = 0;
    p_sys->i_pending_pts = VLC_TICK_INVALID;
    p_sys->b_quit      = false;

    vlc_mutex_init( &p_sys->lock );
    vlc_cond_init(  &p_sys->cond );

    /* --- tell VLC we want mono FL32 @ 16 kHz ---------------------------- */
    p_filter->fmt_in.audio.i_format          = VLC_CODEC_FL32;
    p_filter->fmt_in.audio.i_rate            = ASR_SAMPLE_RATE;
    p_filter->fmt_in.audio.i_physical_channels = AOUT_CHAN_CENTER;
    p_filter->fmt_in.audio.i_channels        = 1;
    aout_FormatPrepare( &p_filter->fmt_in.audio );
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;

    /* --- create shared libvlc variables (read by asr_subsource) --------- */
    vlc_object_t *p_vlc = VLC_OBJECT( vlc_object_instance(p_filter) );
    var_Create( p_vlc, ASR_VAR_TEXT, VLC_VAR_STRING );
    var_Create( p_vlc, ASR_VAR_PTS,  VLC_VAR_INTEGER );
    var_SetString(  p_vlc, ASR_VAR_TEXT, "" );
    var_SetInteger( p_vlc, ASR_VAR_PTS,  0  );

    p_filter->p_sys = p_sys;

    /* --- start background inference thread ------------------------------ */
    if( vlc_clone( &p_sys->thread, InferenceThread, p_filter ) )
    {
        msg_Err( p_filter, "failed to start inference thread" );
        whisper_free( p_sys->p_ctx );
        free( p_sys->p_buf );
        free( p_sys );
        return VLC_EGENERIC;
    }

    static const struct vlc_filter_operations filter_ops =
        { .filter_audio = Process, .close = Close };
    p_filter->ops = &filter_ops;

    msg_Dbg( p_filter, "ASR filter opened (chunk=%ds, rate=%dHz)",
             ASR_CHUNK_SEC, ASR_SAMPLE_RATE );
    return VLC_SUCCESS;
}

/* -------------------------------------------------------------------------
 * Process  – called on the audio thread for every decoded block
 * ---------------------------------------------------------------------- */
static block_t *Process( filter_t *p_filter, block_t *p_block )
{
    filter_sys_t *p_sys     = p_filter->p_sys;
    const float  *p_samples = (const float *)p_block->p_buffer;
    size_t        i_samples = p_block->i_nb_samples;

    /* Record PTS of the very first sample in the current accumulation */
    if( p_sys->i_buf_fill == 0 )
        p_sys->i_buf_pts = p_block->i_pts;

    /* Drain the incoming block into our accumulation buffer, flushing
     * to the worker thread whenever we have a full chunk.              */
    while( i_samples > 0 )
    {
        size_t i_space = ASR_CHUNK_SAMPLES - p_sys->i_buf_fill;
        size_t i_copy  = (i_samples < i_space) ? i_samples : i_space;

        memcpy( p_sys->p_buf + p_sys->i_buf_fill,
                p_samples,
                i_copy * sizeof(float) );

        p_sys->i_buf_fill += i_copy;
        p_samples         += i_copy;
        i_samples         -= i_copy;

        if( p_sys->i_buf_fill >= ASR_CHUNK_SAMPLES )
        {
            /* Hand the full chunk to the worker thread if it is idle */
            vlc_mutex_lock( &p_sys->lock );
            if( p_sys->p_pending == NULL )
            {
                /* Transfer ownership of the buffer */
                p_sys->p_pending     = p_sys->p_buf;
                p_sys->i_pending     = p_sys->i_buf_fill;
                p_sys->i_pending_pts = p_sys->i_buf_pts;

                /* Allocate a fresh accumulation buffer */
                p_sys->p_buf = malloc( ASR_CHUNK_SAMPLES * sizeof(float) );
                if( unlikely(!p_sys->p_buf) )
                {
                    /* OOM: reclaim the pending buffer and drop this chunk */
                    p_sys->p_buf     = p_sys->p_pending;
                    p_sys->p_pending = NULL;
                }
                else
                {
                    vlc_cond_signal( &p_sys->cond );
                }
            }
            /* else: worker is still busy – silently drop this chunk */
            vlc_mutex_unlock( &p_sys->lock );

            p_sys->i_buf_fill = 0;
            p_sys->i_buf_pts  = VLC_TICK_INVALID;
        }
    }

    return p_block; /* pass audio through unchanged */
}

/* -------------------------------------------------------------------------
 * InferenceThread  – runs whisper_full() off the audio thread
 * ---------------------------------------------------------------------- */
static void *InferenceThread( void *p_data )
{
    filter_t     *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys    = p_filter->p_sys;

    char *psz_lang = var_InheritString( p_filter, CFG_PREFIX "language" );
    int   i_threads = var_InheritInteger( p_filter, CFG_PREFIX "threads" );

    vlc_mutex_lock( &p_sys->lock );
    for(;;)
    {
        /* Wait for a chunk or quit signal */
        while( p_sys->p_pending == NULL && !p_sys->b_quit )
            vlc_cond_wait( &p_sys->cond, &p_sys->lock );

        if( p_sys->b_quit )
            break;

        /* Take ownership of the pending chunk */
        float      *p_chunk     = p_sys->p_pending;
        size_t      i_chunk     = p_sys->i_pending;
        vlc_tick_t  i_chunk_pts = p_sys->i_pending_pts;
        p_sys->p_pending = NULL;
        vlc_mutex_unlock( &p_sys->lock );

        /* --- run whisper ------------------------------------------------ */
        struct whisper_full_params wparams =
            whisper_full_default_params( WHISPER_SAMPLING_GREEDY );

        wparams.n_threads       = i_threads;
        wparams.language        = psz_lang ? psz_lang : "auto";
        wparams.translate       = false;
        wparams.single_segment  = false;
        wparams.print_progress  = false;
        wparams.print_realtime  = false;
        wparams.no_timestamps   = false;

        if( whisper_full( p_sys->p_ctx, wparams, p_chunk, (int)i_chunk ) == 0 )
        {
            int n_seg = whisper_full_n_segments( p_sys->p_ctx );
            vlc_object_t *p_vlc = VLC_OBJECT( vlc_object_instance(p_filter) );

            for( int i = 0; i < n_seg; i++ )
            {
                const char *psz_text = whisper_full_get_segment_text( p_sys->p_ctx, i );
                /* whisper timestamps are in centiseconds from chunk start */
                int64_t t0_cs = whisper_full_get_segment_t0( p_sys->p_ctx, i );
                vlc_tick_t i_pts = i_chunk_pts
                                 + VLC_TICK_FROM_MS( t0_cs * 10 );

                var_SetString(  p_vlc, ASR_VAR_TEXT, psz_text );
                var_SetInteger( p_vlc, ASR_VAR_PTS,  i_pts    );

                msg_Dbg( p_filter, "ASR [%"PRId64"ms]: %s",
                         MS_FROM_VLC_TICK(i_pts), psz_text );
            }
        }
        else
        {
            msg_Warn( p_filter, "whisper_full() failed on this chunk" );
        }

        free( p_chunk );
        vlc_mutex_lock( &p_sys->lock );
    }
    vlc_mutex_unlock( &p_sys->lock );

    free( psz_lang );
    return NULL;
}

/* -------------------------------------------------------------------------
 * Close
 * ---------------------------------------------------------------------- */
static void Close( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    /* Signal the worker thread to exit */
    vlc_mutex_lock( &p_sys->lock );
    p_sys->b_quit = true;
    vlc_cond_signal( &p_sys->cond );
    vlc_mutex_unlock( &p_sys->lock );
    vlc_join( p_sys->thread, NULL );

    /* Clean up shared variables */
    vlc_object_t *p_vlc = VLC_OBJECT( vlc_object_instance(p_filter) );
    var_Destroy( p_vlc, ASR_VAR_TEXT );
    var_Destroy( p_vlc, ASR_VAR_PTS  );

    whisper_free( p_sys->p_ctx );
    free( p_sys->p_pending );
    free( p_sys->p_buf );
    free( p_sys );
}
