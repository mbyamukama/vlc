/*****************************************************************************
 * asr_subsource.c : ASR subtitle sub-source for VLC
 *
 * Companion to modules/audio_filter/asr/asr_filter.c.
 * Reads the "asr-caption-text" / "asr-caption-pts" libvlc variables written
 * by the audio filter and emits a subpicture_t each time the text changes,
 * displayed at the bottom of the video like normal subtitles.
 *
 * Enable both modules together:
 *   --audio-filter=asr --sub-source=asr_subsource --asr-model=<path>
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <string.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_subpicture.h>
#include <vlc_text_style.h>
#include <vlc_threads.h>

#define CFG_PREFIX   "asr-"
#define ASR_VAR_TEXT "asr-caption-text"
#define ASR_VAR_PTS  "asr-caption-pts"

/* How long (ms) to keep a caption on screen after it arrives.
 * The audio filter will overwrite it with the next segment anyway. */
#define ASR_DISPLAY_MS  4000

/* -------------------------------------------------------------------------
 * Private state
 * ---------------------------------------------------------------------- */
typedef struct
{
    vlc_mutex_t  lock;
    char        *psz_text;      /* latest recognised text, or NULL  */
    vlc_tick_t   i_pts;         /* PTS when the text was produced   */
    bool         b_changed;     /* true when text was updated       */
} filter_sys_t;

/* -------------------------------------------------------------------------
 * Forward declarations
 * ---------------------------------------------------------------------- */
static int          Open   ( filter_t * );
static void         Close  ( filter_t * );
static subpicture_t *Filter( filter_t *, vlc_tick_t );
static int          OnTextChanged( vlc_object_t *, char const *,
                                   vlc_value_t, vlc_value_t, void * );

/* -------------------------------------------------------------------------
 * Module descriptor
 * ---------------------------------------------------------------------- */
vlc_module_begin()
    set_shortname( N_("ASR Subtitle Source") )
    set_description( N_("Automatic speech recognition subtitle source") )
    set_subcategory( SUBCAT_VIDEO_SUBPIC )
    set_callback_sub_source( Open, 0 )
vlc_module_end()

/* -------------------------------------------------------------------------
 * Open
 * ---------------------------------------------------------------------- */
static int Open( filter_t *p_filter )
{
    filter_sys_t *p_sys = malloc( sizeof(*p_sys) );
    if( unlikely(!p_sys) )
        return VLC_ENOMEM;

    vlc_mutex_init( &p_sys->lock );
    p_sys->psz_text = NULL;
    p_sys->i_pts    = VLC_TICK_INVALID;
    p_sys->b_changed = false;

    p_filter->p_sys = p_sys;

    /* Watch the shared variable written by the audio filter */
    vlc_object_t *p_vlc = VLC_OBJECT( vlc_object_instance(p_filter) );
    var_Create( p_vlc, ASR_VAR_TEXT, VLC_VAR_STRING  );
    var_Create( p_vlc, ASR_VAR_PTS,  VLC_VAR_INTEGER );
    var_AddCallback( p_vlc, ASR_VAR_TEXT, OnTextChanged, p_sys );

    static const struct vlc_filter_operations filter_ops =
        { .source_sub = Filter, .close = Close };
    p_filter->ops = &filter_ops;

    return VLC_SUCCESS;
}

/* -------------------------------------------------------------------------
 * OnTextChanged  – variable callback, called from the inference thread
 * ---------------------------------------------------------------------- */
static int OnTextChanged( vlc_object_t *p_obj, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval,
                          void *p_data )
{
    VLC_UNUSED(psz_var); VLC_UNUSED(oldval);

    filter_sys_t *p_sys = (filter_sys_t *)p_data;
    const char   *psz   = newval.psz_string;

    if( !psz || psz[0] == '\0' )
        return VLC_SUCCESS;

    vlc_mutex_lock( &p_sys->lock );
    free( p_sys->psz_text );
    p_sys->psz_text  = strdup( psz );
    p_sys->i_pts     = (vlc_tick_t)var_GetInteger( p_obj, ASR_VAR_PTS );
    p_sys->b_changed = true;
    vlc_mutex_unlock( &p_sys->lock );

    return VLC_SUCCESS;
}

/* -------------------------------------------------------------------------
 * Filter  – called every video frame; returns a subpicture or NULL
 * ---------------------------------------------------------------------- */
static subpicture_t *Filter( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    vlc_mutex_lock( &p_sys->lock );

    /* Only emit a new subpicture when the text has changed */
    if( !p_sys->b_changed || !p_sys->psz_text )
    {
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    char *psz_text = strdup( p_sys->psz_text );
    p_sys->b_changed = false;
    vlc_mutex_unlock( &p_sys->lock );

    if( unlikely(!psz_text) )
        return NULL;

    /* --- build the subpicture ------------------------------------------ */
    subpicture_t *p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
    {
        free( psz_text );
        return NULL;
    }

    subpicture_region_t *p_region = subpicture_region_NewText();
    if( !p_region )
    {
        subpicture_Delete( p_spu );
        free( psz_text );
        return NULL;
    }

    /* Text content */
    p_region->p_text = text_segment_New( psz_text );
    free( psz_text );

    /* Position: bottom-centre, like normal subtitles */
    p_region->i_align    = SUBPICTURE_ALIGN_BOTTOM;
    p_region->b_absolute = false;
    p_region->b_in_window = false;
    p_region->i_x        = 0;
    p_region->i_y        = 0;
    p_region->fmt.i_sar_num = 1;
    p_region->fmt.i_sar_den = 1;

    vlc_spu_regions_push( &p_spu->regions, p_region );

    /* Timing */
    p_spu->i_start    = date;
    p_spu->i_stop     = date + VLC_TICK_FROM_MS( ASR_DISPLAY_MS );
    p_spu->b_ephemer  = true;   /* replaced by the next segment anyway */
    p_spu->b_subtitle = true;

    return p_spu;
}

/* -------------------------------------------------------------------------
 * Close
 * ---------------------------------------------------------------------- */
static void Close( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    vlc_object_t *p_vlc = VLC_OBJECT( vlc_object_instance(p_filter) );
    var_DelCallback( p_vlc, ASR_VAR_TEXT, OnTextChanged, p_sys );

    free( p_sys->psz_text );
    free( p_sys );
}
