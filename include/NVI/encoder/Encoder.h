#pragma once

#define DllExport __declspec( dllexport )

/*
#if defined (_WIN32)
	#if defined(NaturalVirtualInteractionVideoEncoder_EXPORTS)
		#define  NaturalVirtualInteractionVideoEncoder__declspec(dllexport)
	#else
		#define   NaturalVirtualInteractionVideoEncoder__declspec(dllimport)
	#endif // MyLibrary_EXPORTS
#else // defined (_WIN32)
	#define MYLIB_EXPORT
#endif
*/


extern "C" {
	DllExport int vpx_init(int width, int height);

	DllExport int vpx_encode(const char* rgb_frame, char* encoded, bool force_key_frame);

	DllExport void vpx_cleanup();
}

