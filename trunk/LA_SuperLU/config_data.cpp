#include "config_data.h"
#include "../config.h"

// Some issues with daeConfig and cross compiling using mingw...
void get_from_config(bool& bUseUserSuppliedWorkSpace, real_t& dWorkspaceMemoryIncrement, real_t& dWorkspaceSizeMultiplier, std::string& strReuse)
{
    daeConfig& cfg = daeConfig::GetConfig();
    bUseUserSuppliedWorkSpace = cfg.GetBoolean  ("daetools.superlu.useUserSuppliedWorkSpace",    false);
    dWorkspaceMemoryIncrement = cfg.GetFloat("daetools.superlu.workspaceMemoryIncrement",    1.5);
    dWorkspaceSizeMultiplier  = cfg.GetFloat("daetools.superlu.workspaceSizeMultiplier",     2.0);
    strReuse                  = cfg.GetString("daetools.superlu.factorizationMethod",    std::string("SamePattern"));
}

