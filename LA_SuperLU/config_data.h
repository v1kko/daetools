#include "../Core/definitions.h"

// Some issues with daeConfig and cross compiling using mingw...
void get_from_config(bool& bUseUserSuppliedWorkSpace, real_t& dWorkspaceMemoryIncrement, real_t& dWorkspaceSizeMultiplier, std::string& strReuse);
int  get_numthreads_from_config();


