// dllmain.h : Declaration of module class.

class CDAEToolsCapeOpenModule : public ATL::CAtlDllModuleT< CDAEToolsCapeOpenModule >
{
public :
	DECLARE_LIBID(LIBID_DAEToolsCapeOpenLib)
	DECLARE_REGISTRY_APPID_RESOURCEID(IDR_DAETOOLSCAPEOPEN, "{62296404-EE00-4D61-B213-12CD6CF0DAB7}")
};

extern class CDAEToolsCapeOpenModule _AtlModule;
