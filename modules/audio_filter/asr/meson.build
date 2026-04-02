# ASR (whisper.cpp) audio filter
#
# whisper.cpp must be built separately and its library + headers made
# available.  Set the following meson options (or wrap/subproject) to point
# at the installed tree:
#
#   whisper_dep  – a meson dependency() or declare_dependency() for libwhisper
#
# Quick local build:
#   cd modules/audio_filter/asr/whisper.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build
#   Then pass  -Dwhisper_libdir=<abs_path_to_build>  to VLC's meson setup.

whisper_inc = include_directories('whisper.cpp/include')

# Try pkg-config first, then fall back to find_library
whisper_dep = dependency('whisper', required : false)

if not whisper_dep.found()
    whisper_lib = cc.find_library('whisper',
        dirs : get_option('whisper_libdir'),
        required : false)
    if whisper_lib.found()
        whisper_dep = whisper_lib
    endif
endif

if whisper_dep.found()
    vlc_modules += {
        'name'         : 'asr',
        'sources'      : files('asr_filter.c'),
        'dependencies' : [whisper_dep],
        'include_directories' : whisper_inc,
    }
else
    warning('whisper not found – ASR audio filter will not be built. ' +
            'Create whisper.pc or pass -Dwhisper_libdir=<path>.')
endif
